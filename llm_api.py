"""
llm_api.py
~~~~~~~~~~

Google Generative AI (genai) client functions and LLM interactions.
Encapsulates all direct interactions with the Google Generative AI API.
"""

import math
import os
from logging import Logger
from time import sleep
from typing import Any, List
from pydantic import Field

from groq import Groq

from models import VideoTopics
from utils import enforce_no_additional_properties, truncate
from llm_client_config import LlmClientConfig
from llm_prompts import llm_prompts as prompts

class YtVideoRecommenderLlmConfig(LlmClientConfig):
    """ LLM configuration specific to YouTube video recommender app. """
    classify_videos_batch_size: int = Field(
        default=10,
        gt=1,
        lt=100,
        description="Number of video titles to classify in a single LLM request.",
    )



# ------------------------ LLM client functions ------------------------
def get_llm_config():
    """ Returns the LLM configuration used by the app. """
    if "GROQ_API_KEY" not in os.environ:
        raise EnvironmentError("GROQ_API_KEY environment variable not set.")

    return YtVideoRecommenderLlmConfig(api_key=os.environ["GROQ_API_KEY"])


def create_llm_client(llm_config: YtVideoRecommenderLlmConfig):
    """
    Create and return an initialized Groq client.

    Returns:
        Groq: An authenticated client for the Groq API.
    """
    return Groq(
        api_key=llm_config.api_key,
    )


def classify_videos(
    videos: List[dict],
    llm_client: Any,
    llm_config: YtVideoRecommenderLlmConfig,
    default_topics: List[str],
    logger: Logger,
) -> VideoTopics:
    """
    Classify videos based on their titles into semantic topics using Groq LLM.
    Uses Groq's structured JSON output to ensure reliable parsing.
    """
    logger.info("Started classifying videos by topic using Groq...")

    video_titles = [v["snippet"]["title"] for v in videos]
    topic_pool: set[str] = set(default_topics)
    video_topics = VideoTopics(topics=[])
    batches = int(math.ceil(len(video_titles) / max(1, llm_config.classify_videos_batch_size)))

    for i in range(batches):
        logger.info(f"...classifying videos in batch {i + 1}/{batches}")

        start_idx = i * llm_config.classify_videos_batch_size
        end_idx = min(len(video_titles), start_idx + llm_config.classify_videos_batch_size)
        batch_titles = video_titles[start_idx:end_idx]

        prompt = prompts["classify_videos"]["user"].format(
            video_titles=batch_titles, topics=sorted(topic_pool)
        )

        logger.debug(f"Prompt sent to Groq: {prompt}")

        # Call Groq API with structured output
        json_schema = enforce_no_additional_properties(VideoTopics.model_json_schema())

        response = llm_client.chat.completions.create(
            model=llm_config.model,
            messages=[
                {"role": "system", "content": prompts["classify_videos"]["sys"]},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            #max_completion_tokens=llm_config.max_output_tokens,
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
            logger.warning("No response returned from Groq!")

        sleep(llm_config.politeness_delay_sec)

    logger.info("Done partitioning videos by topic using Groq.")
    return video_topics



def generate_video_search_query(
    topic: str,
    sampled_videos: List[dict],
    target_languages: List[str],
    llm_client: Any,
    llm_config: YtVideoRecommenderLlmConfig,
    logger: Logger,
) -> str:
    """
    Generate a creative YouTube search query for a given topic.

    Uses LLM to generate diverse and targeted search queries based on sampled videos
    from a topic. Supports multilingual output.

    Args:
        topic: Topic category to generate query for
        sampled_videos: List of sample videos from the topic
        target_languages: List of target languages for the search query
        llm_client: LLM client instance
        logger: Logger instance

    Returns:
        str: Generated search query (3-5 words)
    """
    logger.info(
        f"Started generating a YT search query for topic {topic} (using {len(sampled_videos)} videos)..."
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

    response = llm_client.chat.completions.create(
        model=llm_config.model,
        messages=[
            {"role": "system", "content": prompts["generate_search_query"]["sys"]},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
    )

    message = response.choices[0].message
    search_query = message.content.strip()

    logger.info(f'Done generating YT search query: "{search_query}".')
    
    sleep(llm_config.politeness_delay_sec)

    return search_query
