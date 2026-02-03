import os
import random
from logging import Logger
from typing import TYPE_CHECKING

from RecommendationWorkflow import RecommendationWorkflow

if TYPE_CHECKING:
    from youtube_client import YouTubeClient
    from llm_client import LlmClient
    from YoutubeVideosRecommender import YoutubeVideosRecommender


class TopicBasedRecommender(RecommendationWorkflow):
    def run(
        self,
        yt_client: "YouTubeClient",
        llm_client: "LlmClient",
        settings: any,
        logger: Logger,
        recommender: "YoutubeVideosRecommender",
    ) -> None:
        """
        Runs the full video recommendation workflow.
        """
        try:
            videos = yt_client.fetch_liked_videos(settings.num_liked_videos)
            random.shuffle(
                videos
            )  # Shuffle to avoid have similar videos next to each other in collection.

            topic_buckets = recommender.build_topic_buckets(
                videos, llm_client, settings.default_topics
            )

            sampled_videos = recommender.sample_bucket_videos(
                topic_buckets, settings.num_topics, settings.num_videos_topic, logger
            )

            search_queries = recommender.generate_search_queries(
                sampled_videos, settings.target_languages, llm_client
            )
            logger.debug(search_queries)

            result_vids_by_topic = recommender.search_topic_videos(
                search_queries, yt_client, logger, settings
            )

            recommendations_playlist = os.environ["YT_RECOMMENDATIONS_PLAYLIST_ID"]
            prev_recommended_vids = yt_client.fetch_playlist_videos(
                recommendations_playlist, 100
            )

            recommendations = recommender.select_recommended_videos(
                result_vids_by_topic,
                prev_recommended_vids,
                settings.num_recommendations,
                logger,
            )

            yt_client.add_videos_to_playlist(
                recommendations,
                recommendations_playlist,
            )

            yt_client.delete_videos_from_playlist(
                prev_recommended_vids,
                5 * settings.num_recommendations,
            )
        except Exception:
            logger.exception("Recommendation Workflow failed!")
            raise
