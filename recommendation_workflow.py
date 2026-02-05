from abc import ABC, abstractmethod
from logging import Logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from youtube_client import YouTubeClient
    from llm_client import LlmClient
    from youtube_videos_recommender import YoutubeVideosRecommender


class RecommendationWorkflow(ABC):
    """Abstract base for recommendation workflows."""

    @abstractmethod
    def run(
        self,
        yt_client: "YouTubeClient",
        llm_client: "LlmClient",
        settings: any,
        logger: Logger,
        recommender: "YoutubeVideosRecommender",
    ) -> None:
        """
        Runs the recommendation workflow.
        """
        pass
