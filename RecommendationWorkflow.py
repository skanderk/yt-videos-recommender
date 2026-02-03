from abc import ABC, abstractmethod
from logging import Logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from youtube_client import YouTubeClient
    from llm_client import LlmClient
    from YoutubeVideosRecommender import YoutubeVideosRecommender


class RecommendationWorkflow(ABC):
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
