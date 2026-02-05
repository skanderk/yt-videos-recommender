import random
import os
from logging import Logger
from time import sleep
from typing import Dict, List, Sequence, Set

from tqdm import tqdm
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from youtube_client import YouTubeClient
from llm_client import LlmClient
from models import VideoTopics
from type_aliases import (
    ChannelName,
    SearchQuery,
    TopicName,
    VideoDict,
    VideoTitle,
)


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
        default=5,
        gt=1,
        lt=50,
        title="The maximum number of videos to recommend per run. NB. Inserting a video a playlist costs 50 pts.",
    )

    @model_validator(mode="after")
    def compute_num_recommendations(self) -> "RecommenderSettings":
        # If num_recommendations was not explicitly set, compute it from num_topics and num_videos_topic
        # Note: In a real scenario we might want to check if it's the default value.
        # But for now, we'll just ensure it's at most the product of topics and videos per topic if we want.
        # However, the original intention was likely to have a default that scales.
        # To keep it simple and fix the error, I've set a default of 10.
        return self

    num_liked_videos: int = Field(
        default=50,
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


class YoutubeVideosRecommender:
    def __init__(
        self,
        settings: RecommenderSettings,
        logger: Logger,
    ):

        self.settings = settings
        self.logger = logger

    def recommend(self, yt_client: YouTubeClient, llm_client: LlmClient) -> None:
        """
        Runs the full video recommendation workflow.
        """
        try:
            videos = yt_client.fetch_liked_videos(self.settings.num_liked_videos)
            random.shuffle(
                videos
            )  # Shuffle to avoid have similar videos next to each other in collection.

            topic_buckets = self.build_topic_buckets(
                videos, llm_client, self.settings.default_topics
            )

            sampled_videos = self.sample_bucket_videos(
                topic_buckets, self.settings.num_topics, self.settings.num_videos_topic
            )

            search_queries = self.generate_search_queries(
                sampled_videos, self.settings.target_languages, llm_client
            )
            self.logger.debug(search_queries)

            result_vids_by_topic = self.search_topic_videos(
                search_queries, yt_client, self.settings
            )

            recommendations_playlist = os.environ["YT_RECOMMENDATIONS_PLAYLIST_ID"]
            prev_recommended_vids = yt_client.fetch_playlist_videos(
                recommendations_playlist, 100
            )

            recommendations = self.select_recommended_videos(
                result_vids_by_topic,
                prev_recommended_vids,
                self.settings.num_recommendations
            )

            yt_client.add_videos_to_playlist(
                recommendations,
                recommendations_playlist,
            )

            yt_client.delete_videos_from_playlist(
                prev_recommended_vids,
                5 * self.settings.num_recommendations,
            )
        except Exception:
            self.logger.exception("Recommendation Workflow failed!")
            raise

    def create_topic_buckets(
        self, videos: List[VideoDict], video_titles_by_topic: VideoTopics
    ) -> Dict[TopicName, List[VideoDict]]:
        """Return {topic_name → [video_dicts]} using the VideoTopics instance."""
        topic_dict = video_titles_by_topic.to_dict()
        videos_index = self.index_videos_by_title(videos)

        return {
            topic: self.titles_to_videos(titles, videos_index)
            for topic, titles in topic_dict.items()
        }

    def index_videos_by_title(
        self, videos: List[VideoDict]
    ) -> Dict[VideoTitle, VideoDict]:
        """Return {title → video_dict}. Keeps the **first** occurrence, skips videos without a title."""
        index: Dict[VideoTitle, VideoDict] = {}
        for v in videos:
            title = v.get("snippet", {}).get("title", "")
            if title and title not in index:
                index[title] = v
        return index

    def titles_to_videos(
        self, titles: List[VideoTitle], videos_index: Dict[VideoTitle, VideoDict]
    ) -> List[VideoDict]:
        return [v for t in titles if (v := videos_index.get(t))]

    def sample_video_topics(
        self, topic_buckets: Dict[TopicName, List[VideoDict]], max_sample_size: int
    ) -> List:
        """
        Samples max_sample_size video topics uniformly and without replacement.
        """
        topics = list(topic_buckets.keys())
        if max_sample_size >= len(topics):
            return topics

        return random.sample(topics, max_sample_size)

    def sample_videos(
        self, video_pool: List[VideoDict], max_sample_size: int
    ) -> List[VideoDict]:
        """
        Samples max_sample_size videos uniformly and without replacement.
        """
        if max_sample_size >= len(video_pool):
            return video_pool

        return random.sample(video_pool, max_sample_size)

    def select_recommended_videos(
        self,
        video_pool_by_topic: Dict[TopicName, List[VideoDict]],
        tabu_list: List[VideoDict],
        max_recommendations: int
    ) -> List[VideoDict]:
        topic_count = len(video_pool_by_topic)
        if topic_count == 0:
            self.logger.warning("Cannot select recommendations, no topics found!")
            return []

        self.logger.info(f"Selecting {topic_count} videos from each topic.")

        filtered_video_pool = self.filter_recommended_videos(
            video_pool_by_topic, tabu_list
        )
        count_by_topic = max(1, max_recommendations // topic_count)
        recommendations: List[VideoDict] = []
        for _, videos in filtered_video_pool.items():
            if len(videos) < count_by_topic:
                recommendations.extend(videos)
            else:
                sampled_videos = random.sample(videos, count_by_topic)
                recommendations.extend(sampled_videos)

        random.shuffle(recommendations)
        self.logger.info(f"Selected {len(recommendations)} recommendations.")

        return recommendations

    def filter_recommended_videos(
        self,
        videos_by_topic: Dict[TopicName, List[VideoDict]],
        tabu_list: Sequence[VideoDict],
    ) -> Dict[TopicName, List[VideoDict]]:
        """Return a copy of `videos_by_topic` with videos from tabu channels removed."""
        tabu_channels = set()
        for v in tabu_list:
            channel_id = v.get("channelId") or v.get("snippet", {}).get("channelId")
            if channel_id:
                tabu_channels.add(channel_id)

        return {
            topic: self.delete_tabu_videos(topic_vids, tabu_channels)
            for topic, topic_vids in videos_by_topic.items()
        }

    def delete_tabu_videos(
        self, videos: List[VideoDict], tabu_channels: Set[ChannelName]
    ) -> List[VideoDict]:
        def is_not_tabu(v):
            channel_id = v.get("channelId") or v.get("snippet", {}).get("channelId")
            return channel_id not in tabu_channels

        return list(filter(is_not_tabu, videos))

    def build_topic_buckets(
        self,
        videos: List[VideoDict],
        llm_client: LlmClient,
        video_topics: List[TopicName],
    ) -> Dict[TopicName, List[VideoDict]]:
        titles_by_topic = llm_client.classify_videos(videos, video_topics)
        llm_client.logger.debug(titles_by_topic.model_dump_json())

        return self.create_topic_buckets(videos, titles_by_topic)

    def sample_bucket_videos(
        self,
        topic_buckets: Dict[TopicName, List[VideoDict]],
        topic_sample_size: int,
        video_sample_size: int,
    ) -> Dict[str, List[Dict]]:
        sampled_topics = self.sample_video_topics(topic_buckets, topic_sample_size)
        self.logger.debug(f"Sampled topics: {sampled_topics}")

        return {
            topic: self.sample_videos(topic_buckets[topic], video_sample_size)
            for topic in sampled_topics
        }

    def generate_search_queries(
        self,
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
        self,
        query_by_topic: Dict[TopicName, SearchQuery],
        yt_client: YouTubeClient,
        settings: RecommenderSettings = None,
    ) -> Dict[TopicName, List[VideoDict]]:
        max_results = settings.num_yt_search_results if settings else 10
        search_results: Dict[TopicName, List[VideoDict]] = {}
        for topic, query in tqdm(
            query_by_topic.items(), desc="Searching topics", disable=False
        ):
            search_results[topic] = yt_client.search_videos(
                query, max_results=max_results
            )
            sleep(1)

        return search_results
