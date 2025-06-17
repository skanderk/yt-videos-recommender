# YouTube LLM-Based Videos Recommender

Suggest fresh videos based **only** on your _Liked Videos_ playlist, while skipping channels you have placed in a *tabu* list.  
The recommender uses the **YouTube Data API v3** (OAuth 2.0) to read your likes and the **Google Generative AI (Gemini) API** to classify liked videos and generate YT search queries to discover new videos.

* **Language** Python ≥ 3.10  
* **Dependency manager** [Poetry]  
* **License** Apache 2.0

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)  
2. [Clone & Install](#2-clone--install)  
3. [Authenticate Once](#3-authenticate-once)  
4. [Run the Recommender](#4-run-the-recommender)  
5. [Schedule a Daily Cron (Optional)](#5-schedule-a-daily-cron-optional)  
6. [Configuration Files](#6-configuration-files)  
7. [Contributing](#7-contributing)  
8. [License](#8-license)

---

## 1  Prerequisites

### 1.1 Create a Google Cloud project & enable APIs

1. Sign in to **<https://console.cloud.google.com/>** and click **“New project”**.  
   Name it **`yt-videos-recommender`** (or anything you like).
2. In the new project open **“APIs & Services → Library”** and **Enable**  
   - **YouTube Data API v3**  
   - **Generative AI API** (Gemini)  
     <sub>The console may list it as “Generative AI API for Google AI Studio” or “AI Generative Language API”.</sub>
3. Navigate to **“OAuth consent screen”** ➜ choose **_External_**, fill the minimal fields, press **Save & Continue** through the remaining pages.
4. Go to **“Credentials → + Create credentials → OAuth client ID”**  
   * **Application type** Desktop app  
   * **Name** `yt-recommender-desktop`  
   Click **Create** and download the JSON.
5. Rename the file to **`oauth_client_secret.json`** and place it in the repository root.  
   > **Note** GitHub’s secret-scanning rejects anything resembling real OAuth secrets, so this file is **never committed**.  
6. Still under **Credentials**, click **“+ Create credentials → API key.”**  
   Copy the key—you will store it in `.env` as `GEMINI_API_KEY`.

### 1.2 Local tools

* Git ≥ 2.35 (SSH configured)  
* Python 3.12 runtime (system or via pyenv)  
* Poetry (install via `pipx install poetry`)

---

## 2  Clone & Install

```bash
git clone git@github.com:skanderk/yt-videos-recommender.git
cd yt-videos-recommender

cp .env.example .env          # fill in your own values
poetry install
