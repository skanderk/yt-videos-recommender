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
import math
import os
import random
from logging import Logger
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Dict, List, Tuple, Set, Sequence

import dotenv
from google import genai
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.genai import Client as LLMClient
from google.genai.types import GenerateContentConfig
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource as YouTubeClient
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel

# ------------------------ Recommender configs ------------------------
LIKED_VIDEOS_TO_PULL = 1000  # The maximum number of liked videos to pull from YT. Used to make recommendations.
MAX_YT_SEARCH_RESULTS = 15  # The maximum number of videos to return in a YT search. Increase for more diversity.
TOPIC_SAMPLE_SIZE = 5  # The maximum number of video topics to be recommended.
VIDEO_SAMPLE_SIZE = 3  # The maximum number of videos to be sampled from each topic to generate search queries.
RECOMMENDATIONS_COUNT = 5  # The maximum number of videos to recommend per run. NB. Inserting a video a playlist costs 50 pts.

# ------------------------ YouTube API Scopes read/write permissions required by recommender ------------------------
OAUTH_SECRETS_FILE = "oauth.json"  # File exported from Google console.
OAUTH_TOKEN_FILE = (
    "token.json"  # File containing the current OAuth token and a refresh token.
)


YT_API_SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

# ------------------------ Google GenAI configs ------------------------
LLM = "gemini-2.0-flash"

# --> Customize to your preferred topics, if required.
# Some of these topics were inspired by : https://developers.google.com/youtube/v3/docs/search/list
# The Recommender is multilingual, though English is internally used as the Lingua Franka for topic names.
DEFAULT_VIDEO_TOPICS = [
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
    "Sports",
    "Technology",
    "Tourism",
    "Tv Shows",
    "Vehicles",
]


CLASSIFY_VIDEOS_SYS_PROMPT = """
You are a video content classifier. 
You receive a list of YouTube video titles and return a JSON dictionary mapping each title to one of 
the provided topics based on semantic meaning and your internal knowledge.
"""

CLASSIFY_VIDEOS_USER_PROMPT = """
Classify each of the following titles into *one and only one* topic.


Video titles:
{video_titles}  # batch titles in groups

Pick up the topics from the following list. Create a new topic if it is *necessary* and only if it is necessary.

topics:
{topics} # topic names
"""

CLASSIFY_VIDEOS_BATCH_SIZE = 20

# --> Customize to your preferred languages.
SEARCH_TARGET_LANGUAGES = ["english", "french"]

GEN_SEARCH_QUERY_SYS_PROMPT = """You are a creative search queries generator.
You receive a list of YouTube Videos all of the same topic and return a search query that allows to retrieve videos 
somewhat related to the ones you have received. I want the search queries to allow me to discover 
new and diverse content on YouTube. 
Your response must be a *single string* representing the search query you have generated.  
"""

GEN_SEARCH_QUERY_USER_PROMPT = """
Generate a single YouTube search query given the list of videos below covering this given topic *topic*. The search query must be in one of the languages
specified below (selected randomly). Use the video title and description of each video to *norrow* its topic from *topic* to a *video_subtopic*. Pick up randomly one of the *video_subtopic*s, 
reword it, then generate a YouTube search query in one of and only one of *languages*. The search query must have at least three 
words and at most five words. 

topic:
{topic}

videos:
{videos}

languages:
{languages}
"""

# ------------------------ Misc configs ------------------------
LOG_FILE = None  # Log file path. Set to None to log to console.
TWEEKING = False  # Set to True to log messages useful for tweeking the recommender.


# ------------------------ Recommender functions ------------------------
def get_oauth_credentials() -> Credentials:
    """
    Gets an OAuth2 credentials (access token) from Google, if required, then creates a YouTube API client.
    """
    if not os.path.exists(OAUTH_TOKEN_FILE):
        oauth_flow = InstalledAppFlow.from_client_secrets_file(
            OAUTH_SECRETS_FILE, scopes=YT_API_SCOPES
        )
        credentials = oauth_flow.run_local_server(
            port=9090,
            prompt="consent",
            access_type="offline",
            include_granted_scopes="true",
        )

        Path(OAUTH_TOKEN_FILE).write_text(credentials.to_json())
    else:
        credentials = Credentials.from_authorized_user_file(OAUTH_TOKEN_FILE)
        if credentials.expired:
            credentials.refresh(GoogleAuthRequest())
            Path(OAUTH_TOKEN_FILE).write_text(credentials.to_json())

    return credentials


def create_youtube_client(oauth_credentials: Credentials) -> YouTubeClient:
    """
    Returns A YouTube API client.
    """
    return build("youtube", "v3", credentials=oauth_credentials, cache_discovery=False)


def create_llm_client() -> LLMClient:
    """
    Returns an initialized LLM client.
    """
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def fetch_playlist_items(
    playlist_id: str,
    max_videos: int,
    yt_client: YouTubeClient,
    logger: Logger,
) -> List[Dict]:
    """
    Fetches up to <max_videos> from playlist <plylist_id>
    """
    fetched = []

    try:
        parts = "id,snippet,contentDetails"

        request = yt_client.playlistItems().list(
            part=parts, playlistId=playlist_id, maxResults=min(max_videos, 50)
        )

        while request and len(fetched) < max_videos:
            response = request.execute()
            fetched.extend(item for item in response["items"])
            request = yt_client.playlistItems().list_next(request, response)
    except Exception as e:
        logger.error(f"Failed to fetch (some) videos with exception {e}")

    fetch_videos = fetched[:max_videos]

    return fetch_videos


def fetch_liked_videos(
    max_videos: int, yt_client: YouTubeClient, logger: Logger
) -> List[Dict]:
    """
    Fetches up to <max_videos> YouTube videos that current user liked.
    """
    logger.info(f"Started fetching up to {max_videos} liked videos from YouTube...")
    liked_videos = fetch_playlist_items("LL", max_videos, yt_client, logger)
    logger.info(f"Done fetching  {len(liked_videos)} liked videos from YouTube.")

    return liked_videos


def delete_old_llm_videos(
    videos: List[Dict],
    videos_to_keep_count: int,
    yt_client: YouTubeClient,
    logger: Logger,
) -> None:
    logger.info(
        f"Started deleting old recommended videos, keeping {videos_to_keep_count} recommendations..."
    )

    try:
        videos_count = len(videos)
        logger.info(f"...fetched {videos_count} from recommendations playlist.")
        if videos_count > videos_to_keep_count:
            # Delete at most 10 videos, to avoid exceeding YouTube API quota (10000 pt, 50 pt per delete request)
            videos_to_delete_count = min(10, videos_count - videos_to_keep_count)
            logger.info(
                f"...deleting {videos_to_delete_count} older recommendations from playlist."
            )

            for vid in list(reversed(videos))[:videos_to_delete_count]:
                playlistitem_id = vid["id"]
                logger.info(f"...deleting item {playlistitem_id}")
                request = yt_client.playlistItems().delete(id=playlistitem_id)
                request.execute()
                sleep(1)

    except Exception as e:
        logger.warning(f"Failed to delete older recommendations with error {e}")

    logger.info("Done deleting old recommended videos.")


# Classes Topic and VideoTopics serve as the response schema for
# the LLM “classify-videos” prompt.
class Topic(BaseModel):
    """A single topic grouping a list of YouTube video titles."""

    name: str
    videos: List[str]


class VideoTopics(BaseModel):
    """
    A collection of ``Topic`` instances.
    """

    topics: List[Topic]

    def to_dict(self) -> Dict[str, List[str]]:
        return {tp.name: tp.videos for tp in self.topics}

    def get_topics(self) -> List[str]:
        return [t.name for t in self.topics]

    @staticmethod
    def from_dict(d: Dict[str, List[str]]) -> "VideoTopics":
        topics = [Topic(name=name, videos=vids) for name, vids in d.items()]
        return VideoTopics(topics=topics)

    def __add__(self, other: "VideoTopics") -> "VideoTopics":
        """
        Merges the topics of two ``VideoTopics``. Immutable, returns a new ``VideoTopics`` with the
        merged topics.

        Handy when you run an LLM on multiple chunks of video titles and need
        to fold the per-chunk topic maps into a single, consolidated result.
        """
        if not isinstance(other, VideoTopics):
            return NotImplemented

        merged = self.to_dict()
        for ot in other.topics:
            if ot.name in merged:
                merged[ot.name] = list(
                    dict.fromkeys(merged[ot.name] + ot.videos)
                )  # Deduplicate
            else:
                merged[ot.name] = ot.videos.copy()

        return VideoTopics.from_dict(merged)


def classify_videos(
    videos: List[dict], llm_client: LLMClient, logger: Logger
) -> VideoTopics:
    """
    Classifies videos based on their titles into semantic topics using an LLM.

    Returns
    -------
    VideoTopics
        Merged topic mapping for all input videos.
    """
    logger.info("Started classifying videos by topic using an LLM...")

    video_titles = [v["snippet"]["title"] for v in videos]

    topic_pool: set[str] = set(DEFAULT_VIDEO_TOPICS)
    gencontent_config = GenerateContentConfig(
        system_instruction=CLASSIFY_VIDEOS_SYS_PROMPT,
        temperature=0,
        response_schema=VideoTopics,
        response_mime_type="application/json",
    )

    video_topics = VideoTopics(topics=[])
    batches = int(math.ceil(len(video_titles) / max(1, CLASSIFY_VIDEOS_BATCH_SIZE)))

    for i in range(0, batches):
        logger.info(f"...classifying videos in batch {i + 1}/{batches}")

        start_idx = i * CLASSIFY_VIDEOS_BATCH_SIZE
        end_idx = min(len(video_titles), start_idx + CLASSIFY_VIDEOS_BATCH_SIZE)

        prompt = CLASSIFY_VIDEOS_USER_PROMPT.format(
            video_titles=video_titles[start_idx:end_idx], topics=sorted(topic_pool)
        )

        response = llm_client.models.generate_content(
            model=LLM,
            contents=prompt,
            config=gencontent_config,
        )

        if response.parsed:
            video_topics = video_topics + response.parsed
        else:
            logger.warning("Failed to get a response from the LLM!")

        # Feed default and newly-discovered video topics back to the LLM.
        topic_pool |= set(video_topics.get_topics())
        sleep(1)

    logger.info("Done partionning videos by topic using LLM.")
    return video_topics


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


def generate_video_search_query(
    topic: str,
    sampled_videos: List[Dict],
    target_languages: List[str],
    llm_client,
    logger: Logger,
) -> str:
    logger.info(
        f"Started generating a YT search query for topic {topic} (using {len(sampled_videos)} videos)..."
    )

    # --> Customize temperature and top_p to balance diversity/relevance of generated search query.
    gencontent_config = GenerateContentConfig(
        system_instruction=GEN_SEARCH_QUERY_SYS_PROMPT, temperature=1.0, top_p=0.7
    )

    prompt_videos = [
        {
            "video_title": v["snippet"]["title"],
            "video_description": v["snippet"]["description"],
        }
        for v in sampled_videos
    ]

    prompt = GEN_SEARCH_QUERY_USER_PROMPT.format(
        videos=prompt_videos, languages=target_languages, topic=topic
    )

    response = llm_client.models.generate_content(
        model=LLM,
        contents=prompt,
        config=gencontent_config,
    )

    search_query = response.text
    logger.info(f'Done generating YT search query: "{search_query}".')

    return search_query


def search_youtube(
    query: str, max_results: int, yt_client, logger: Logger
) -> List[Dict]:
    logger.info(f"Searching Youtube with query: '{query}'(max_results{max_results})...")

    try:
        response = (
            yt_client.search()
            .list(
                part="snippet",
                q=query,
                type="video",
                maxResults=max_results,
                order="relevance",
                videoDefinition="high",
                videoDuration="long",
            )
            .execute()
        )

        videos = []
        for item in response["items"]:
            videoId = item["id"]["videoId"]
            snippet = item["snippet"]
            videos.append(
                {
                    "videoId": videoId,
                    "title": snippet["title"],
                    "channelId": snippet["channelId"],
                    "publishedAt": snippet["publishedAt"],
                }
            )

        logger.info(f"Done searching Youtube, got {len(videos)} results.")
        return videos

    except HttpError as error:
        logger.error(f"Youtube search failed with error: {error}!")
        return []


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


def add_to_recommendation_playlist(
    videos: List[Dict], playlist_id: str, yt_client, logger: Logger
) -> None:
    logger.info(
        f"Started adding {len(videos)} reommendations to your LLM recommendations playlist..."
    )

    added_count = 0
    for video in videos:
        try:
            logger.info(f"...adding video {video['title']}-{video['videoId']}")

            yt_client.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "position": 0,
                        "resourceId": {
                            "kind": "youtube#video",
                            "videoId": video["videoId"],
                        },
                    }
                },
            ).execute()
            added_count += 1
            sleep(1)

        except HttpError as e:
            logger.error(e)

    logger.info(
        f"Done adding {added_count} reommendations to your LLM recommendations playlist..."
    )


def run_recommendation_workflow(
    yt_client: YouTubeClient, llm_client: LLMClient, logger: Logger
) -> None:
    """
    Runs the full video recommendation workflow.
    """
    try:
        videos = fetch_liked_videos(LIKED_VIDEOS_TO_PULL, yt_client, logger)
        random.shuffle(
            videos
        )  # Shuffle to avoid have similar videos next to each other in collection.

        topic_buckets = build_topic_buckets(videos, llm_client, logger)

        sampled_videos = sample_bucket_videos(
            topic_buckets, TOPIC_SAMPLE_SIZE, VIDEO_SAMPLE_SIZE, logger
        )

        search_queries = generate_search_queries(
            sampled_videos, SEARCH_TARGET_LANGUAGES, llm_client, logger
        )
        logger.debug(search_queries)

        result_vids_by_topic = search_topic_videos(search_queries, yt_client, logger)

        recommendations_playlist = os.environ["YT_RECOMMENDATIONS_PLAYLIST_ID"]
        prev_rcommended_vids = fetch_playlist_items(
            recommendations_playlist, 100, yt_client, logger
        )

        recommendations = select_recommended_videos(
            result_vids_by_topic, prev_rcommended_vids, RECOMMENDATIONS_COUNT, logger
        )

        add_to_recommendation_playlist(
            recommendations,
            recommendations_playlist,
            yt_client,
            logger,
        )

        delete_old_llm_videos(
            prev_rcommended_vids,
            5 * RECOMMENDATIONS_COUNT,
            yt_client,
            logger,
        )
    except Exception:
        logger.exception("Recommendation Workflow failed!")
        raise


def build_topic_buckets(
    videos: List[Dict], llm_client: LLMClient, logger: Logger
) -> Dict[str, List[str]]:
    titles_by_topic = classify_videos(videos, llm_client, logger)
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

    clock_start, logger = setup()
    logger.info("--> Started making YouTube video recommendations ...")

    youtube_client, llm_client = create_clients()
    run_recommendation_workflow(youtube_client, llm_client, logger)

    # Finalizing
    run_time_secs = perf_counter() - clock_start
    logger.info(
        f"<-- Done making YouTube video recommendations in {run_time_secs} sec."
    )


def setup() -> Tuple[float, Logger]:
    """
    Setup chores: creates a logger, loads env. variables...

    Returns
    -------
        * Clock start time.
        * A configured logger.
    """
    clock_start = perf_counter()
    logger = create_logger()
    load_environment(logger)

    return clock_start, logger


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
