import logging
import os
from logging import Logger
from time import perf_counter
from typing import Tuple

import dotenv
from rich.logging import RichHandler

from google_oauth_client import GoogleOAuthClient
from youtube_client import YouTubeClient
from llm_client import (
    LlmClient,
    YtVideoRecommenderLlmConfig,
)

from youtube_videos_recommender import YoutubeVideosRecommender, RecommenderSettings
from topic_based_recommender import TopicBasedRecommender


# ------------------------ OAuth configuration ------------------------
OAUTH_SECRETS_FILE = "oauth.json"
OAUTH_TOKEN_FILE = "token.json"

YT_API_SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

# ------------------------ Misc configs ------------------------
TWEEKING = False  # Set to True to log messages useful for tweeking the recommender.


def main() -> None:
    """Drives the end-to-end YouTube recommendation job."""

    settings, clock_start, logger = setup()
    logger.info("--> Started making YouTube video recommendations ...")

    youtube_client, llm_client = create_clients(logger)

    workflow = TopicBasedRecommender()
    recommender = YoutubeVideosRecommender(workflow, settings, logger)

    recommender.run(youtube_client, llm_client)

    # Finalizing
    run_time_secs = perf_counter() - clock_start
    logger.info(
        f"<-- Done making YouTube video recommendations in {run_time_secs} sec."
    )


def setup() -> Tuple[RecommenderSettings, float, Logger]:
    """
    Setup chores: creates a logger, loads env. variables...

    Returns
    -------
        * Settings of the recommendation system.
        * Clock start time.
        * A configured logger.
    """
    recommender_settings = RecommenderSettings()
    clock_start = perf_counter()
    logger = create_logger()
    load_environment(logger)

    return recommender_settings, clock_start, logger


def create_logger():
    """Creates a preconfigured logger."""
    logger = logging.getLogger("yt_recommender")
    if TWEEKING:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    rich_handler = RichHandler(show_time=False, show_path=False, rich_tracebacks=True)
    logging.basicConfig(
        level=log_level,
        encoding="utf8",
        format="%(asctime)s[%(levelname)s]: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[rich_handler],
    )

    return logger


def load_environment(logger: Logger):
    """
    Loads the environment required by recommender and checks that all evironment variables are set.
    """
    dotenv.load_dotenv(override=True)

    all_set = True
    if os.environ.get("GOOGLE_API_KEY", "") == "":
        logger.critical("""
                        Env. variable GOOGLE_API_KEY is not set.
                        Please create a new Google Cloud project and API key:
                        https://developers.google.com/workspace/guides/create-credentials
                        """)
        all_set = False

    if os.environ.get("GROQ_API_KEY", "") == "":
        logger.critical("""
                        Env. variable GROQ_API_KEY is not set.
                        Please create a new Groq Cloud project and API key:
                        https://console.groq.com/home
                        """)
        all_set = False

    if not os.environ.get("OAUTH_CLIENT_ID") or not os.environ.get("OAUTH_CLIENT_SECRET"):
        logger.critical("""
                        OAuth credententials are not set (env. variables OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET).
                        Please configure OAuth for your Google console project:
                        https://developers.google.com/workspace/guides/create-credentials
                        """)
        all_set = False

    if not os.environ.get("YT_RECOMMENDATIONS_PLAYLIST_ID"):
        logger.critical("""
                        Env. variable YT_RECOMMENDATIONS_PLAYLIST_ID is not set.
                        Please create a new Youtube list (called e.g. LLM Recommendations) and assign its id to
                        this variable.You can make this playlist private.
                        """)
        all_set = False

    if not all_set:
        exit(-1)


def create_clients(logger: Logger) -> Tuple[YouTubeClient, LlmClient]:
    """
    Authenticates and creates the YouTube Data API client and LLM client.
    """
    oauth_client = GoogleOAuthClient(
        secrets_file=OAUTH_SECRETS_FILE, token_file=OAUTH_TOKEN_FILE
    )
    outh_credentials = oauth_client.get_credentials(scopes=YT_API_SCOPES)
    yt_client = YouTubeClient(outh_credentials, logger)

    llm_client = LlmClient(create_llm_config(), logger)

    return yt_client, llm_client


def create_llm_config():
    """Returns the LLM configuration used by the app."""
    if "GROQ_API_KEY" not in os.environ:
        raise EnvironmentError("GROQ_API_KEY environment variable not set.")

    return YtVideoRecommenderLlmConfig(api_key=os.environ["GROQ_API_KEY"])


if __name__ == "__main__":
    main()
