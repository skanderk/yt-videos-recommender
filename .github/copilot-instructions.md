# Copilot Instructions for YouTube LLM-Based Videos Recommender

## Overview
This project is a YouTube video recommender that suggests videos based on the user's liked videos while avoiding specific channels. It utilizes the YouTube Data API v3 and the Google Generative AI (Gemini) API for video classification and search query generation.

## Architecture
- **Main Components**:
  - `recommend_yt_videos.py`: The core script that implements the recommendation logic.
  - `oauth.json` and `token.json`: Handle OAuth 2.0 authentication for accessing the YouTube Data API.
  - `pyproject.toml`: Manages project dependencies using Poetry.

- **Data Flow**:
  1. User authenticates via OAuth, generating `oauth.json` and `token.json`.
  2. The recommender fetches liked videos using the YouTube Data API.
  3. The Gemini API classifies these videos and generates new search queries.

## Developer Workflows
- **Setup**:
  - Clone the repository and install dependencies:
    ```bash
    git clone git@github.com:skanderk/yt-videos-recommender.git
    cd yt-videos-recommender
    cp .env.example .env          # fill in your own values
    poetry install
    ```

- **Running the Recommender**:
  - Execute the main script:
    ```bash
    poetry run python recommend_yt_videos.py
    ```

- **Testing**:
  - Ensure to write tests for new features. Use `pytest` for running tests.

## Project Conventions
- **Dependency Management**: Use Poetry for managing dependencies. Always update `pyproject.toml` when adding new packages.
- **Configuration Files**: Store sensitive information in `.env` and never commit `oauth_client_secret.json`.

## Integration Points
- **YouTube Data API**: Ensure the API is enabled in your Google Cloud project.
- **Gemini API**: Store the API key in the `.env` file as `GEMINI_API_KEY`.

## Examples
- To add a new feature, follow the existing patterns in `recommend_yt_videos.py` and ensure to update the README with any new setup instructions.