from typing import List
from time import sleep

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from logging import Logger

from type_aliases import SearchQuery, VideoDict


class YouTubeClient:
    """A Client for interacting with the YouTube Data API."""

    def __init__(self, credentials: Credentials, logger: Logger):
        self.client = build(
            "youtube", "v3", credentials=credentials, cache_discovery=False
        )
        self.logger = logger

    def fetch_playlist_videos(
        self,
        playlist_id: str,
        max_videos: int,
    ) -> List[VideoDict]:
        """
        Fetch up to `max_videos` items from a YouTube playlist.

        Args:
            playlist_id: YouTube playlist ID
            max_videos: Maximum number of videos to fetch

        Returns:
            List of playlist items (videos)
        """
        if max_videos <= 0:
            return []

        fetched = []

        try:
            parts = "id,snippet,contentDetails"

            request = self.client.playlistItems().list(
                part=parts, playlistId=playlist_id, maxResults=min(max_videos, 50)
            )

            while request and len(fetched) < max_videos:
                response = request.execute()
                fetched.extend(item for item in response["items"])
                request = self.client.playlistItems().list_next(request, response)
        except Exception as e:
            self.logger.error(f"Failed to fetch (some) videos with exception {e}")

        return fetched[:max_videos]

    def fetch_liked_videos(self, max_videos: int) -> List[VideoDict]:
        """
        Fetch up to `max_videos` YouTube videos that the current user has liked.
        Uses the "LL" (Liked List) special playlist ID.

        Args:
            max_videos: Maximum number of liked videos to fetch

        Returns:
            List of liked video items
        """
        self.logger.info(
            f"Started fetching up to {max_videos} liked videos from YouTube..."
        )

        if max_videos <= 0:
            liked_videos = []
        else:
            liked_videos = self.fetch_playlist_videos("LL", max_videos)

        self.logger.info(
            f"Done fetching  {len(liked_videos)} liked videos from YouTube."
        )

        return liked_videos

    def search_videos(self, query: SearchQuery, max_results: int) -> List[VideoDict]:
        """
        Search YouTube for videos matching a query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of video search results with metadata
        """
        if max_results <= 0:
            return []

        self.logger.info(
            f"Searching Youtube with query: '{query}'(max_results{max_results})..."
        )
        try:
            response = (
                self.client.search()
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

            self.logger.info(f"Done searching Youtube, got {len(videos)} results.")
            return videos

        except HttpError as error:
            self.logger.error(f"Youtube search failed with error: {error}!")
            return []

    def add_videos_to_playlist(self, videos: List[VideoDict], playlist_id: str) -> None:
        """ Add videos to a YouTube playlist."""
        self.logger.info(
            f"Started adding {len(videos)} videos to playlist {playlist_id}...")

        added_count = 0
        for video in videos:
            try:
                self.logger.info(f"...adding video {video['title']}-{video['videoId']}")

                self.client.playlistItems().insert(
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
                self.logger.error(e)

        self.logger.info(
            f"Done adding {added_count} videos to playlist {playlist_id}."
        )

    def delete_videos_from_playlist(
        self,
        videos: List[VideoDict],
        videos_to_keep_count: int,
    ) -> None:
        """Delete old videos from a YouTube playlist, keeping only the most recent ones."""
        self.logger.info(
            f"Started deleting old videos from playlist, keeping {videos_to_keep_count} most recent."
        )

        try:
            videos_count = len(videos)
            self.logger.info(f"...fetched {videos_count} videos from playlist.")
            if videos_count > videos_to_keep_count:
                # Delete at most 10 videos, to avoid exceeding YouTube API quota (10000 pt, 50 pt per delete request)
                videos_to_delete_count = min(10, videos_count - videos_to_keep_count)
                self.logger.info(
                    f"..deleting {videos_to_delete_count} old videos from playlist."
                )

                for vid in list(reversed(videos))[:videos_to_delete_count]:
                    playlistitem_id = vid["id"]
                    self.logger.info(f"...deleting item {playlistitem_id}")
                    request = self.client.playlistItems().delete(id=playlistitem_id)
                    request.execute()
                    sleep(1)

        except Exception as e:
            self.logger.warning(
                f"Failed to delete old videos from playlist with error: {e}"
            )

        self.logger.info("Done deleting old videos from playlist.")
