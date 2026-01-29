# LLM prompts for video classification
CLASSIFY_VIDEOS_SYS_PROMPT = """
You are a diligent video content classifier. 
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

# Dictionary of LLM prompts
llm_prompts = {
    "classify_videos": {"sys": CLASSIFY_VIDEOS_SYS_PROMPT, "user": CLASSIFY_VIDEOS_USER_PROMPT},
    "generate_search_query": {"sys": GEN_SEARCH_QUERY_SYS_PROMPT, "user": GEN_SEARCH_QUERY_USER_PROMPT},
}
