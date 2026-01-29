
from pydantic import BaseModel
from typing import Dict, List


class Topic(BaseModel):
    """A single topic grouping a list of YouTube video titles."""

    name: str
    videos: List[str]


class VideoTopics(BaseModel):
    """
    A collection of ``Topic`` instances.
    """

    topics: List[Topic]

    def to_dict(self) -> Dict[str, List[str]]:
        return {tp.name: tp.videos for tp in self.topics}

    def get_topics(self) -> List[str]:
        return [t.name for t in self.topics]

    @staticmethod
    def from_dict(d: Dict[str, List[str]]) -> "VideoTopics":
        topics = [Topic(name=name, videos=vids) for name, vids in d.items()]
        return VideoTopics(topics=topics)

    def __add__(self, other: "VideoTopics") -> "VideoTopics":
        """
        Merges the topics of two ``VideoTopics``. Immutable, returns a new ``VideoTopics`` with the
        merged topics.

        Handy when you run an LLM on multiple chunks of video titles and need
        to fold the per-chunk topic maps into a single, consolidated result.
        """
        if not isinstance(other, VideoTopics):
            return NotImplemented

        merged = self.to_dict()
        for ot in other.topics:
            if ot.name in merged:
                merged[ot.name] = list(
                    dict.fromkeys(merged[ot.name] + ot.videos)
                )  # Deduplicate
            else:
                merged[ot.name] = ot.videos.copy()

        return VideoTopics.from_dict(merged)