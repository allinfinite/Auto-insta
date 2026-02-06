# Auto-Insta

AI-powered Instagram content generation and autoposter. Combines a Flask dashboard, background posting daemon, and Telegram bot with Venice AI to create a full content pipeline: topic-based questions, voice/text responses, AI-generated images and captions, and scheduled Instagram publishing.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Dashboard (Flask)                       │
│  Manage queue, topics, character config, generated content   │
│  http://localhost:5556                                       │
└──────────────┬──────────────────────────────┬───────────────┘
               │ SQLite (shared)              │
┌──────────────▼──────────┐   ┌───────────────▼──────────────┐
│     poster.py           │   │      generator.py             │
│  Background daemon      │   │  Telegram bot + Venice AI     │
│  Publishes queued posts │   │  Content generation pipeline  │
│  via Instagram API      │   │  Question → Response → Post   │
└─────────────────────────┘   └──────────────────────────────┘
```

### Components

- **`dashboard.py`** — Flask web UI for managing everything: posting queue, topics, character identity, generated content review, settings, and service monitoring
- **`poster.py`** — Background daemon that checks the queue every N seconds and publishes due posts via the Instagram API (instagrapi). Handles retries, story sharing, and feed-to-story resharing
- **`generator.py`** — Telegram bot (python-telegram-bot) + Venice AI content generation. Sends topic questions, processes text/voice responses, generates captions and images, sends previews with inline approval buttons

## Features

### Content Generation Pipeline
1. Bot picks a topic (round-robin) and generates an engaging question via LLM
2. You respond with text or a voice memo (auto-transcribed)
3. AI generates an Instagram caption using your brand voice
4. AI generates matching media (image, infographic, or reel)
5. Bot sends a preview with inline buttons: Queue, Post Now, Redo, Cancel
6. Approved content enters the posting queue

### Dashboard Generation
Content can also be generated directly from the web dashboard:
- Pick a topic, write your response, choose media type
- Optionally auto-queue after generation
- Generator bot processes it in the background (~15s)

### Media Types
- **Image** — Photorealistic images via `flux-2-pro` with optional reference image for character likeness
- **Infographic** — Clean, shareable graphics via `nano-banana-pro` (1080x1350)
- **Reel** — AI-generated video from a still image via `wan-2.5-preview-image-to-video`

### Posting Features
- Scheduled posting with drag-to-reorder queue
- Auto-share posts to story
- Automatic feed-to-story resharing throughout the day
- Retry logic for failed posts
- Instagram challenge/2FA handling

### Character Config
- **Reference image** — Upload a photo for character likeness in generated images
- **Physical description** — Text description for image prompts
- **Style descriptors** — Lighting, photography style, aesthetic
- **Camera preferences** — Lens, depth of field, camera model
- **Brand voice** — Tone and personality for caption generation
- **Default hashtags** — Auto-included in generated content
- **Infographic style** — Visual preferences for infographic generation

## Setup

### Prerequisites
- Python 3.10+
- `ffmpeg` (for voice memo transcription): `brew install ffmpeg`
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- A [Venice AI](https://venice.ai) API key
- Instagram account credentials

### Installation

```bash
git clone https://github.com/allinfinite/Auto-insta.git
cd Auto-insta
pip install -r requirements.txt
pip install "python-telegram-bot[job-queue]"
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

| Variable | Description |
|----------|-------------|
| `INSTAGRAM_USERNAME` | Instagram account username |
| `INSTAGRAM_PASSWORD` | Instagram account password |
| `DASHBOARD_PASSWORD` | Web dashboard login password |
| `DASHBOARD_PORT` | Dashboard port (default: 5556) |
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | Your Telegram user ID (for receiving questions) |
| `TELEGRAM_ALLOWED_USERS` | Comma-separated user IDs allowed to use the bot |
| `VENICE_API_KEY` | Venice AI API key |

To find your Telegram user ID, message [@userinfobot](https://t.me/userinfobot) on Telegram.

### Venice AI Models

| Purpose | Default Model | Config Key |
|---------|--------------|------------|
| Chat/LLM | `llama-3.3-70b` | `VENICE_CHAT_MODEL` |
| Photo images | `flux-2-pro` | `VENICE_IMAGE_MODEL` |
| Infographics | `nano-banana-pro` | `VENICE_INFOGRAPHIC_MODEL` |
| Video (img2vid) | `wan-2.5-preview-image-to-video` | `VENICE_VIDEO_MODEL` |
| Transcription | `nvidia/parakeet-tdt-0.6b-v3` | `VENICE_TRANSCRIPTION_MODEL` |

## Usage

### Start the Dashboard

```bash
python dashboard.py
```

Open http://localhost:5556 and log in with your dashboard password.

### Start the Poster Daemon

```bash
python poster.py
```

On first run, it will prompt for Instagram login (and 2FA if enabled). The session is cached in `ig_session.json` for subsequent runs.

### Start the Generator Bot

```bash
python generator.py
```

Then message your bot on Telegram:

### Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Show help |
| `/generate [topic]` | Get a question (random or specific topic) |
| `/skip` | Cancel current pending question |
| `/status` | Show generator stats |
| `/topics` | List available topics |
| `/type image\|reel\|infographic` | Set media type for next generation |

### Inline Preview Buttons

When content is generated, the bot sends a preview with:
- **Queue** — Add to posting queue (scheduled after last pending post)
- **Post Now** — Schedule for immediate publishing
- **Redo All** — Regenerate caption + image
- **Redo Caption** — Regenerate just the caption
- **Redo Image** — Regenerate just the image
- **Cancel** — Discard the session

### Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | Stats, service status, upcoming/recent posts, logs |
| **Queue** | Drag-to-reorder pending posts, post now, edit, delete |
| **New Post** | Manually create a post with media upload |
| **History** | Browse posted/failed posts, retry, share to story |
| **Topics** | CRUD for content topics, generate button per topic |
| **Character** | Reference image, physical description, brand voice, style |
| **Generated** | Review AI-generated content, approve/preview/delete |
| **Settings** | Instagram creds, posting config, generator config, services |

## Database

SQLite with WAL mode (`instagram_autoposter.db`). All three services share the same database.

### Tables

| Table | Purpose |
|-------|---------|
| `posts` | Posting queue with scheduling, status tracking, media paths |
| `settings` | Key-value configuration store |
| `post_log` | Audit log of all post actions |
| `story_reshares` | Tracks feed-to-story reshares (7-day dedup) |
| `topics` | Content topics with descriptions and question prompts |
| `character_config` | Character identity settings for AI generation |
| `generation_sessions` | Tracks the full question → response → content pipeline |

## Running as Services (macOS)

The poster and generator can be managed as launchd services. Set the service labels in `.env`:

```
POSTER_SERVICE_LABEL=com.instagram.autoposter
GENERATOR_SERVICE_LABEL=com.instagram.generator
```

The dashboard provides start/stop/restart controls for the poster daemon in Settings.

## Tech Stack

- **Backend**: Python, Flask, SQLite
- **Instagram API**: [instagrapi](https://github.com/subzeroid/instagrapi)
- **Telegram**: [python-telegram-bot](https://python-telegram-bot.org/) (Bot API)
- **AI**: [Venice AI](https://venice.ai) (OpenAI-compatible API)
- **Image Gen**: Flux 2 Pro, Nano Banana Pro
- **Video Gen**: Wan 2.5 (image-to-video)
- **Transcription**: Nvidia Parakeet TDT
- **Audio**: ffmpeg (OGG to MP3 conversion)

## License

MIT
