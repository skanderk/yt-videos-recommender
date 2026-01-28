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
from typing import Any, Dict, List, Tuple, Set, Sequence

import dotenv
from google.genai import Client as LLMClient
from googleapiclient.discovery import Resource as YouTubeClient
from pydantic import  Field
from pydantic_settings import BaseSettings

# Import YouTube API functions
from youtube_api import (
    get_oauth_credentials,
    create_youtube_client,
    fetch_playlist_items,
    fetch_liked_videos,
    search_youtube,
    add_to_recommendation_playlist,
    delete_old_llm_videos,
)

# Import LLM API functions and configuration
from llm_api import (
    VideoTopics,
    create_llm_client,
    classify_videos,
    generate_video_search_query,
)


# ------------------------ Recommender settings ------------------------
class RecommenderSettings(BaseSettings):
    """
    Settings of the YouTube videos recommender. Customize per your needs and likings.
    """

    num_topics: int = Field(
        default=5,
        gt=1,
        lt=20,
        title="The maximum number of videos to return in a YT search. Increase for more diversity.",
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
        gt=0,
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
        default=["english", "french"], title="Languages of recommended videos"
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
LOG_FILE = None  # Log file path. Set to None to log to console.
TWEEKING = False  # Set to True to log messages useful for tweeking the recommender.


# ------------------------ Recommender functions ------------------------


# Classes Topic and VideoTopics serve as the response schema for
# the LLM “classify-videos” prompt.
def create_topic_buckets(
    videos: List[Dict], video_titles_by_topic: VideoTopics
) -> Dict[str, List[Dict]]:
    topic_dict = video_titles_by_topic.to_dict()
    videos_index = index_videos_by_title(videos)

    return {
        topic: titles_to_videos(titles, videos_index)
        for topic, titles in topic_dict.items()
    }


def index_videos_by_title(videos: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return {title → video_dict}. Keeps the **first** occurrence, skips videos without a title."""
    index: Dict[str, Dict[str, Any]] = {}
    for v in videos:
        title = v.get("snippet", {}).get("title")
        if title and title not in index:
            index[title] = v
    return index


def titles_to_videos(titles: List[str], videos_index: Dict) -> List[Dict]:
    return [v for t in titles if (v := videos_index.get(t))]


def sample_video_topics(
    topic_buckets: Dict[str, List[Dict]], max_sample_size: int
) -> List:
    """
    Samples max_sample_size video topics uniformly and without replacement.
    """
    topics = list(topic_buckets.keys())
    if max_sample_size >= len(topics):
        return topics

    return random.sample(topics, max_sample_size)


def sample_videos(video_pool: List[Dict], max_sample_size: int) -> List:
    """
    Samples max_sample_size videos uniformly and without replacement.
    """
    if max_sample_size >= len(video_pool):
        return video_pool

    return random.sample(video_pool, max_sample_size)


def select_recommended_videos(
    video_pool_by_topic: Dict[str, List[Dict]],
    tabu_list: List[Dict],
    max_recommendations: int,
    logger: Logger,
) -> List[Dict]:
    topic_count = len(video_pool_by_topic)
    if topic_count == 0:
        logger.warning("Cannot select recommendations, no topics found!")
        return []

    logger.info(f"Selecting {topic_count} videos from each topic.")

    filtered_video_pool = filter_recommended_videos(video_pool_by_topic, tabu_list)
    count_by_topic = max(1, max_recommendations // topic_count)
    recommendations: List[Dict] = []
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
    videos_by_topic: Dict[str, List[Dict]], tabu_list: Sequence[Dict]
) -> Dict[str, List[Dict]]:
    """Return a copy of `videos_by_topic` with videos from tabu channels removed."""
    tabu_channels = set([v["snippet"]["channelId"] for v in tabu_list])

    return {
        topic: delete_tabu_videos(topic_vids, tabu_channels)
        for topic, topic_vids in videos_by_topic.items()
    }


def delete_tabu_videos(videos: List[Dict], tabu_channels: Set[str]) -> List[Dict]:
    return list(filter(lambda v: v["channelId"] not in tabu_channels, videos))



def run_recommendation_workflow(
    yt_client: YouTubeClient,
    llm_client: LLMClient,
    settings: RecommenderSettings,
    logger: Logger,
) -> None:
    """
    Runs the full video recommendation workflow.
    """
    try:
        videos = fetch_liked_videos(settings.num_liked_videos, yt_client, logger)
        random.shuffle(
            videos
        )  # Shuffle to avoid have similar videos next to each other in collection.

        topic_buckets = build_topic_buckets(
            videos, llm_client, settings.default_topics, logger
        )

        sampled_videos = sample_bucket_videos(
            topic_buckets, settings.num_topics, settings.num_videos_topic, logger
        )

        search_queries = generate_search_queries(
            sampled_videos, settings.target_languages, llm_client, logger
        )
        logger.debug(search_queries)

        result_vids_by_topic = search_topic_videos(search_queries, yt_client, logger)

        recommendations_playlist = os.environ["YT_RECOMMENDATIONS_PLAYLIST_ID"]
        prev_rcommended_vids = fetch_playlist_items(
            recommendations_playlist, 100, yt_client, logger
        )

        recommendations = select_recommended_videos(
            result_vids_by_topic,
            prev_rcommended_vids,
            settings.num_recommendations,
            logger,
        )

        add_to_recommendation_playlist(
            recommendations,
            recommendations_playlist,
            yt_client,
            logger,
        )

        delete_old_llm_videos(
            prev_rcommended_vids,
            5 * settings.num_recommendations,
            yt_client,
            logger,
        )
    except Exception:
        logger.exception("Recommendation Workflow failed!")
        raise


def build_topic_buckets(
    videos: List[Dict], llm_client: LLMClient, video_topics: List[str], logger: Logger
) -> Dict[str, List[str]]:
    titles_by_topic = classify_videos(videos, llm_client, video_topics, logger)
    logger.debug(titles_by_topic.model_dump_json())

    return create_topic_buckets(videos, titles_by_topic)


def sample_bucket_videos(
    topic_buckets: Dict[str, List[Dict]],
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
    videos_by_topic: Dict[str, List[Dict]],
    target_languages: List[str],
    llm_client: LLMClient,
    logger: Logger,
) -> Dict[str, str]:
    search_queries: Dict[str, str] = {}
    for topic, videos in videos_by_topic.items():
        search_query = generate_video_search_query(
            topic, videos, target_languages, llm_client, logger
        )
        search_queries[topic] = search_query

    return search_queries


def search_topic_videos(
    query_by_topic: Dict[str, str], yt_client: YouTubeClient, logger: Logger
) -> Dict[str, List[Dict]]:
    search_results: Dict[str, List[Dict]] = {}
    for topic, query in query_by_topic.items():
        search_results[topic] = search_youtube(
            query, max_results=10, yt_client=yt_client, logger=logger
        )
        sleep(1)

    return search_results


# ------------------------ Entry-point function and its helpers ------------------------
def main() -> None:
    """Drives the end-to-end YouTube recommendation job."""

    settings, clock_start, logger = setup()
    logger.info("--> Started making YouTube video recommendations ...")

    youtube_client, llm_client = create_clients()
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

    logging.basicConfig(
        filename=LOG_FILE,
        level=log_level,
        encoding="utf8",
        format="%(asctime)s[%(levelname)s]: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
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


def create_clients() -> Tuple[YouTubeClient, LLMClient]:
    """
    Authenticates to YouTube and creates API clients
    """
    outh_credentials = get_oauth_credentials()
    yt_client = create_youtube_client(outh_credentials)
    llm_client = create_llm_client()

    return yt_client, llm_client


if __name__ == "__main__":
    main()
