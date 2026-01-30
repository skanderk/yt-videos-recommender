"""
recommend_yt_videos.py
~~~~~~~~~~~~~~~~~~~~~~

CLI command that recommends **diverse new YouTube videos** by leveraging an
LLM.

Workflow
--------
1. Pull the user’s liked videos.
2. Classify each video by topic.
3. Sample a subset of topics and videos.
4. Generate search queries from the samples.
5. Search YouTube and pick promising results.
6. Add the selected videos to the “LLM Recommendations” playlist.

See :func:`run_recommendation_workflow` for implementation details.

Author
------
Skander Kort  <https://skanderkort.com>
"""

import logging
import os
import random
from logging import Logger
from time import perf_counter, sleep
from typing import Dict, List, Sequence, Set, Tuple

import dotenv
from rich.logging import RichHandler
from pydantic import Field
from pydantic_settings import BaseSettings
from tqdm import tqdm

from google_oauth_client import GoogleOAuthClient
from youtube_client import YouTubeClient

from llm_client import (
    LlmClient,
    YtVideoRecommenderLlmConfig,
)

from models import VideoTopics
from type_aliases import (
    ChannelName,
    SearchQuery,
    TopicName,
    VideoDict,
    VideoTitle,
)


# ------------------------ OAuth configuration ------------------------
OAUTH_SECRETS_FILE = "oauth.json"
OAUTH_TOKEN_FILE = "token.json"

YT_API_SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]


# ------------------------ Recommender settings ------------------------
class RecommenderSettings(BaseSettings):
    """
    Settings of the YouTube videos recommender. Customize per your needs and likings.
    """

    num_topics: int = Field(
        default=10,
        gt=1,
        lt=20,
        title="The maximum number of topics to sample. Increase for more diversity.",
    )

    num_videos_topic: int = Field(
        default=1,
        gt=0,
        lt=10,
        title="The maximum number of videos to be sampled from each topic to generate search queries.",
    )

    num_recommendations: int = Field(
        default_factory=lambda data: data["num_topics"] * data["num_videos_topic"],
        gt=1,
        lt=50,
        title="The maximum number of videos to recommend per run. NB. Inserting a video a playlist costs 50 pts.",
    )

    num_liked_videos: int = Field(
        default=200,
        gt=1,
        lt=500000,
        title="The maximum number of liked videos to pull from YT. Used to make recommendations.",
    )

    num_yt_search_results: int = Field(
        default=15,
        gt=2,
        le=50,
        title="The maximum number of videos to return in a YT search. Increase for more diversity.",
    )

    target_languages: list[str] = Field(
        default=["english", "french", "arabic"], title="Languages of recommended videos"
    )

    default_topics: list[str] = Field(
        default=[
            "Business",
            "Food",
            "Gaming",
            "Health",
            "Movies",
            "Music",
            "Pets",
            "Philosophy",
            "Politics",
            "Religion",
            "Science",
            "Mathematics",
            "Physics",
            "Sports",
            "Technology",
            "AI",
            "Programming",
            "Tourism",
            "Tv Shows",
            "Vehicles",
            "News",
        ],
        title="""Customize to your preferred topics, if required.
        Some of these topics were inspired by : https://developers.google.com/youtube/v3/docs/search/list
        The Recommender is multilingual, though English is internally used as the Lingua Franka for topic names.
""",
    )


# ------------------------ Misc configs ------------------------
TWEEKING = False  # Set to True to log messages useful for tweeking the recommender.


# ------------------------ Recommender functions ------------------------


# Classes Topic and VideoTopics serve as the response schema for
# the LLM “classify-videos” prompt.
def create_topic_buckets(
    videos: List[VideoDict], video_titles_by_topic: VideoTopics
) -> Dict[TopicName, List[VideoDict]]:
    """Return {topic_name → [video_dicts]} using the VideoTopics instance."""
    topic_dict = video_titles_by_topic.to_dict()
    videos_index = index_videos_by_title(videos)

    return {
        topic: titles_to_videos(titles, videos_index)
        for topic, titles in topic_dict.items()
    }


def index_videos_by_title(videos: List[VideoDict]) -> Dict[VideoTitle, VideoDict]:
    """Return {title → video_dict}. Keeps the **first** occurrence, skips videos without a title."""
    index: Dict[VideoTitle, VideoDict] = {}
    for v in videos:
        title = v.get("snippet", {}).get("title", "")
        if title and title not in index:
            index[title] = v
    return index


def titles_to_videos(
    titles: List[VideoTitle], videos_index: Dict[VideoTitle, VideoDict]
) -> List[VideoDict]:
    return [v for t in titles if (v := videos_index.get(t))]


def sample_video_topics(
    topic_buckets: Dict[TopicName, List[VideoDict]], max_sample_size: int
) -> List:
    """
    Samples max_sample_size video topics uniformly and without replacement.
    """
    topics = list(topic_buckets.keys())
    if max_sample_size >= len(topics):
        return topics

    return random.sample(topics, max_sample_size)


def sample_videos(video_pool: List[VideoDict], max_sample_size: int) -> List[VideoDict]:
    """
    Samples max_sample_size videos uniformly and without replacement.
    """
    if max_sample_size >= len(video_pool):
        return video_pool

    return random.sample(video_pool, max_sample_size)


def select_recommended_videos(
    video_pool_by_topic: Dict[TopicName, List[VideoDict]],
    tabu_list: List[VideoDict],
    max_recommendations: int,
    logger: Logger,
) -> List[VideoDict]:
    topic_count = len(video_pool_by_topic)
    if topic_count == 0:
        logger.warning("Cannot select recommendations, no topics found!")
        return []

    logger.info(f"Selecting {topic_count} videos from each topic.")

    filtered_video_pool = filter_recommended_videos(video_pool_by_topic, tabu_list)
    count_by_topic = max(1, max_recommendations // topic_count)
    recommendations: List[VideoDict] = []
    for topic, videos in filtered_video_pool.items():
        if len(videos) < count_by_topic:
            recommendations.extend(videos)
        else:
            sampled_videos = random.sample(videos, count_by_topic)
            recommendations.extend(sampled_videos)

    random.shuffle(recommendations)
    logger.info(f"Selected {len(recommendations)} recommendations.")

    return recommendations


def filter_recommended_videos(
    videos_by_topic: Dict[TopicName, List[VideoDict]], tabu_list: Sequence[VideoDict]
) -> Dict[TopicName, List[VideoDict]]:
    """Return a copy of `videos_by_topic` with videos from tabu channels removed."""
    tabu_channels = set([v["snippet"]["channelId"] for v in tabu_list])

    return {
        topic: delete_tabu_videos(topic_vids, tabu_channels)
        for topic, topic_vids in videos_by_topic.items()
    }


def delete_tabu_videos(
    videos: List[VideoDict], tabu_channels: Set[ChannelName]
) -> List[VideoDict]:
    return list(filter(lambda v: v["channelId"] not in tabu_channels, videos))


def run_recommendation_workflow(
    yt_client: YouTubeClient,
    llm_client: LlmClient,
    settings: RecommenderSettings,
    logger: Logger,
) -> None:
    """
    Runs the full video recommendation workflow.
    """
    try:
        videos = yt_client.fetch_liked_videos(settings.num_liked_videos)
        random.shuffle(
            videos
        )  # Shuffle to avoid have similar videos next to each other in collection.

        topic_buckets = build_topic_buckets(videos, llm_client, settings.default_topics)

        sampled_videos = sample_bucket_videos(
            topic_buckets, settings.num_topics, settings.num_videos_topic, logger
        )

        search_queries = generate_search_queries(
            sampled_videos, settings.target_languages, llm_client
        )
        logger.debug(search_queries)

        result_vids_by_topic = search_topic_videos(search_queries, yt_client, logger)

        recommendations_playlist = os.environ["YT_RECOMMENDATIONS_PLAYLIST_ID"]
        prev_rcommended_vids = yt_client.fetch_playlist_videos(
            recommendations_playlist, 100
        )

        recommendations = select_recommended_videos(
            result_vids_by_topic,
            prev_rcommended_vids,
            settings.num_recommendations,
            logger,
        )

        yt_client.add_videos_to_playlist(
            recommendations,
            recommendations_playlist,
        )

        yt_client.delete_videos_from_playlist(
            prev_rcommended_vids,
            5 * settings.num_recommendations,
        )
    except Exception:
        logger.exception("Recommendation Workflow failed!")
        raise


def build_topic_buckets(
    videos: List[VideoDict], llm_client: LlmClient, video_topics: List[TopicName]
) -> Dict[TopicName, List[VideoDict]]:
    titles_by_topic = llm_client.classify_videos(videos, video_topics)
    llm_client.logger.debug(titles_by_topic.model_dump_json())

    return create_topic_buckets(videos, titles_by_topic)


def sample_bucket_videos(
    topic_buckets: Dict[TopicName, List[VideoDict]],
    topic_sample_size: int,
    video_sample_size: int,
    logger: Logger,
) -> Dict[str, List[Dict]]:
    sampled_topics = sample_video_topics(topic_buckets, topic_sample_size)
    logger.debug(f"Sampled topics: {sampled_topics}")

    return {
        topic: sample_videos(topic_buckets[topic], video_sample_size)
        for topic in sampled_topics
    }


def generate_search_queries(
    videos_by_topic: Dict[TopicName, List[VideoDict]],
    target_languages: List[str],
    llm_client: LlmClient,
) -> Dict[TopicName, SearchQuery]:
    search_queries: Dict[TopicName, SearchQuery] = {}
    for topic, videos in tqdm(
        videos_by_topic.items(), desc="Generating search queries", disable=False
    ):
        search_query = llm_client.generate_video_search_query(
            topic, videos, target_languages
        )
        search_queries[topic] = search_query

    return search_queries


def search_topic_videos(
    query_by_topic: Dict[TopicName, SearchQuery],
    yt_client: YouTubeClient,
    logger: Logger,
) -> Dict[TopicName, List[VideoDict]]:
    search_results: Dict[TopicName, List[VideoDict]] = {}
    for topic, query in tqdm(
        query_by_topic.items(), desc="Searching topics", disable=False
    ):
        search_results[topic] = yt_client.search_videos(query, max_results=10)
        sleep(1)

    return search_results


# ------------------------ Entry-point function and its helpers ------------------------
def main() -> None:
    """Drives the end-to-end YouTube recommendation job."""

    settings, clock_start, logger = setup()
    logger.info("--> Started making YouTube video recommendations ...")

    youtube_client, llm_client = create_clients(logger)
    run_recommendation_workflow(youtube_client, llm_client, settings, logger)

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
    if os.environ["GOOGLE_API_KEY"] == "":
        logger.critical("""
                        Env. variable GOOGLE_API_KEY is not set. 
                        Please create a new Google Cloud project and API key: 
                        https://developers.google.com/workspace/guides/create-credentials
                        """)
        all_set = False

    if os.environ["GROQ_API_KEY"] == "":
        logger.critical("""
                        Env. variable GROQ_API_KEY is not set. 
                        Please create a new Groq Cloud project and API key: 
                        https://console.groq.com/home
                        """)

    if not os.environ["OAUTH_CLIENT_ID"] or not os.environ["OAUTH_CLIENT_SECRET"]:
        logger.critical("""
                        OAuth credententials are not set (env. variables OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET).
                        Please configure OAuth for your Google console project:
                        https://developers.google.com/workspace/guides/create-credentials
                        """)
        all_set = False

    if not os.environ["YT_RECOMMENDATIONS_PLAYLIST_ID"]:
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
