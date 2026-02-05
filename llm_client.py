import math
from logging import Logger
from time import sleep
from typing import List

from tqdm import tqdm
from pydantic import Field
from groq import Groq

from llm_client_config import LlmClientConfig
from llm_prompts import llm_prompts as prompts
from models import VideoTopics
from utils import enforce_no_additional_properties, truncate


class YtVideoRecommenderLlmConfig(LlmClientConfig):
    """LLM configuration specific to YouTube video recommender app."""

    classify_videos_batch_size: int = Field(
        default=10,
        gt=1,
        lt=100,
        description="Number of video titles to classify in a single LLM request.",
    )


class LlmClient:
    """A client for interacting with the LLM API for YouTube video recommendation tasks."""

    def __init__(self, config: YtVideoRecommenderLlmConfig, logger: Logger):
        self.config = config
        self.logger = logger
        self.client = Groq(api_key=config.api_key)

    def classify_videos(
        self, videos: List[dict], default_topics: List[str]
    ) -> VideoTopics:
        """
        Classify videos based on their titles into semantic topics using Groq LLM.
        Uses Groq's structured JSON output to ensure reliable parsing.
        """
        self.logger.info("Started classifying videos by topic using Groq...")

        video_titles = [v["snippet"]["title"] for v in videos]
        topic_pool: set[str] = set(default_topics)
        video_topics = VideoTopics(topics=[])
        batches = int(
            math.ceil(
                len(video_titles) / max(1, self.config.classify_videos_batch_size)
            )
        )

        for i in tqdm(range(batches), desc="Classifying video batches", disable=False):
            self.logger.info(f"...classifying videos in batch {i + 1}/{batches}")

            start_idx = i * self.config.classify_videos_batch_size
            end_idx = min(
                len(video_titles), start_idx + self.config.classify_videos_batch_size
            )
            batch_titles = video_titles[start_idx:end_idx]

            prompt = prompts["classify_videos"]["user"].format(
                video_titles=batch_titles, topics=sorted(topic_pool)
            )

            self.logger.debug(f"Prompt sent to Groq: {prompt}")

            # Call Groq API with structured output
            json_schema = enforce_no_additional_properties(
                VideoTopics.model_json_schema()
            )

            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": prompts["classify_videos"]["sys"]},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                # max_completion_tokens=self.config.max_output_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "VideoTopics",
                        "strict": True,
                        "schema": json_schema,
                    },
                },
            )

            if response and response.choices:
                structured_output = response.choices[0].message.content

                video_topics = video_topics + VideoTopics.model_validate_json(
                    structured_output
                )
                # Update topic pool for next batch
                topic_pool |= set(video_topics.get_topics())
            else:
                self.logger.warning("No response returned from Groq!")

            sleep(self.config.politeness_delay_sec)

        self.logger.info("Done partitioning videos by topic using Groq.")
        return video_topics

    def generate_video_search_query(
        self,
        topic: str,
        sampled_videos: List[dict],
        target_languages: List[str],
    ) -> str:
        """
        Generate a creative YouTube search query for a given topic.

        Uses LLM to generate diverse and targeted search queries based on sampled videos
        from a topic. Supports multilingual output.

        Args:
            topic: Topic category to generate query for
            sampled_videos: List of sample videos from the topic
            target_languages: List of target languages for the search query

        Returns:
            str: Generated search query (3-5 words)
        """
        self.logger.info(
            f"Started generating a YT search query for topic '{topic}' (using {len(sampled_videos)} videos)..."
        )

        prompt_videos = [
            {
                "video_title": v["snippet"]["title"],
                "video_description": truncate(v["snippet"]["description"]),
            }
            for v in sampled_videos
        ]

        prompt = prompts["generate_search_query"]["user"].format(
            videos=prompt_videos, languages=target_languages, topic=topic
        )

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": prompts["generate_search_query"]["sys"]},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
        )

        message = response.choices[0].message
        search_query = message.content.strip()

        self.logger.info(f'Done generating YT search query: "{search_query}".')

        sleep(self.config.politeness_delay_sec)

        return search_query
