"""
youtube_api.py
~~~~~~~~~~~~~~

YouTube Data API v3 client functions and OAuth authentication handlers.
Encapsulates all direct interactions with the YouTube Data API.
"""

import os
from logging import Logger
from pathlib import Path
from time import sleep
from typing import Dict, List

from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource as YouTubeClient
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# OAuth configuration
OAUTH_SECRETS_FILE = "oauth.json"
OAUTH_TOKEN_FILE = "token.json"

YT_API_SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]


def get_oauth_credentials() -> Credentials:
    """
    Gets an OAuth2 credentials (access token) from Google, if required.
    Creates a new token or refreshes an existing one.
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
    Create and return a YouTube Data API v3 client.
    """
    return build("youtube", "v3", credentials=oauth_credentials, cache_discovery=False)


def fetch_playlist_items(
    playlist_id: str,
    max_videos: int,
    yt_client: YouTubeClient,
    logger: Logger,
) -> List[Dict]:
    """
    Fetch up to `max_videos` items from a YouTube playlist.
    
    Args:
        playlist_id: YouTube playlist ID
        max_videos: Maximum number of videos to fetch
        yt_client: YouTube API client
        logger: Logger instance
        
    Returns:
        List of playlist items (videos)
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

    return fetched[:max_videos]


def fetch_liked_videos(
    max_videos: int, yt_client: YouTubeClient, logger: Logger
) -> List[Dict]:
    """
    Fetch up to `max_videos` YouTube videos that the current user has liked.
    Uses the "LL" (Liked List) special playlist ID.
    
    Args:
        max_videos: Maximum number of liked videos to fetch
        yt_client: YouTube API client
        logger: Logger instance
        
    Returns:
        List of liked video items
    """
    logger.info(f"Started fetching up to {max_videos} liked videos from YouTube...")
    liked_videos = fetch_playlist_items("LL", max_videos, yt_client, logger)
    logger.info(f"Done fetching  {len(liked_videos)} liked videos from YouTube.")

    return liked_videos


def search_youtube(
    query: str, max_results: int, yt_client: YouTubeClient, logger: Logger
) -> List[Dict]:
    """
    Search YouTube for videos matching a query.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        yt_client: YouTube API client
        logger: Logger instance
        
    Returns:
        List of video search results with metadata
    """
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


def add_to_recommendation_playlist(
    videos: List[Dict], playlist_id: str, yt_client: YouTubeClient, logger: Logger
) -> None:
    """
    Add a list of videos to a YouTube playlist.
    
    Args:
        videos: List of videos to add (each must have 'videoId' and 'title')
        playlist_id: Target playlist ID
        yt_client: YouTube API client
        logger: Logger instance
    """
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


def delete_old_llm_videos(
    videos: List[Dict],
    videos_to_keep_count: int,
    yt_client: YouTubeClient,
    logger: Logger,
) -> None:
    """
    Delete old recommendation videos from the playlist, keeping only the most recent.
    Respects YouTube API quota by deleting at most 10 videos per call.
    
    Args:
        videos: List of playlist items to consider for deletion
        videos_to_keep_count: Number of videos to keep (newest)
        yt_client: YouTube API client
        logger: Logger instance
    """
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
