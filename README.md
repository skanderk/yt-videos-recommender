# YouTube Video Recommender

A minimal proof-of-concept that recommends videos from a the liked videos playlist using a Large Language Model
using the YouTube Data API v3.  
Written in **Python 3.10** and managed with **Poetry**.

## Features
* Pulls most-recent playlist items, filters them by topic, channel and tabu list
* Generates a daily recommendation list (console or cron-friendly)
* Caches OAuth tokens (`token.json`) and refreshes automatically

## Quick start

```bash
poetry install
poetry run python recommend_yt_videos.py
