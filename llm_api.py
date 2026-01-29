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

from groq import Groq

from models import VideoTopics
from utils import enforce_no_additional_properties, truncate


# LLM model configuration
LLM = "openai/gpt-oss-120b"  # Groq model to use for all LLM calls

POLITENESS_DELAY_SEC = 10  # Delay between LLM calls to avoid rate limits

# LLM prompts for video classification
CLASSIFY_VIDEOS_SYS_PROMPT = """
You are a video content classifier. 
You receive a list of YouTube video titles and return a JSON dictionary mapping each title to one of 
the provided topics based on semantic meaning and your internal knowledge.
"""

CLASSIFY_VIDEOS_USER_PROMPT = """
Classify each of the following titles into *one and only one* topic.


Video titles:
{video_titles}  # batch titles in groups

Pick up the topics from  list DEFAULT_TOPICS if you find a topic in this list that describes accurately the video. 
Create a new topic if you are not able to find a precise topic in DEFAULT_TOPICS.
Do not create new topics unless necessary.

DEFAULT_TOPICS:
{topics} # topic names
"""

CLASSIFY_VIDEOS_BATCH_SIZE = (
    5  # Number of video titles to classify in a single LLM call.
)

# LLM prompts for search query generation
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


# ------------------------ LLM client functions ------------------------
def create_llm_client():
    """
    Create and return an initialized Groq client.

    Returns:
        Groq: An authenticated client for the Groq API.
    """
    return Groq(
        api_key=os.environ["GROQ_API_KEY"],
    )


def classify_videos(
    videos: List[dict],
    llm_client: Any,
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
    batches = int(math.ceil(len(video_titles) / max(1, CLASSIFY_VIDEOS_BATCH_SIZE)))

    for i in range(batches):
        logger.info(f"...classifying videos in batch {i + 1}/{batches}")

        start_idx = i * CLASSIFY_VIDEOS_BATCH_SIZE
        end_idx = min(len(video_titles), start_idx + CLASSIFY_VIDEOS_BATCH_SIZE)
        batch_titles = video_titles[start_idx:end_idx]

        prompt = CLASSIFY_VIDEOS_USER_PROMPT.format(
            video_titles=batch_titles, topics=sorted(topic_pool)
        )

        logger.debug(f"Prompt sent to Groq: {prompt}")

        # Call Groq API with structured output
        json_schema = enforce_no_additional_properties(VideoTopics.model_json_schema())

        response = llm_client.chat.completions.create(
            model=LLM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
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

        sleep(POLITENESS_DELAY_SEC)

    logger.info("Done partitioning videos by topic using Groq.")
    return video_topics



def generate_video_search_query(
    topic: str,
    sampled_videos: List[dict],
    target_languages: List[str],
    llm_client: Any,
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

    prompt = GEN_SEARCH_QUERY_USER_PROMPT.format(
        videos=prompt_videos, languages=target_languages, topic=topic
    )

    response = llm_client.chat.completions.create(
        model=LLM,
        messages=[
            {"role": "system", "content": GEN_SEARCH_QUERY_SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
    )

    message = response.choices[0].message
    search_query = message.content.strip()

    logger.info(f'Done generating YT search query: "{search_query}".')
    
    sleep(POLITENESS_DELAY_SEC)

    return search_query

