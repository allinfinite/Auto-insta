"""
Instagram Content Generator — Telegram bot + Venice AI content generation daemon.
Sends topic-based questions via Telegram, processes user responses (text/voice),
generates Instagram content (image/reel/infographic) with captions, and queues for posting.
Usage: python3 generator.py
"""

import asyncio
import base64
import json
import logging
import os
import random
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "instagram_autoposter.db"
UPLOADS_DIR = BASE_DIR / "uploads"
LOG_FILE = BASE_DIR / "generator.log"

UPLOADS_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ALLOWED_USERS = {int(x) for x in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",") if x.strip()}

VENICE_API_KEY = os.getenv("VENICE_API_KEY", "")
VENICE_BASE = "https://api.venice.ai/api/v1"
VENICE_CHAT_MODEL = os.getenv("VENICE_CHAT_MODEL", "llama-3.3-70b")
VENICE_IMAGE_MODEL = os.getenv("VENICE_IMAGE_MODEL", "flux-2-pro")
VENICE_INFOGRAPHIC_MODEL = os.getenv("VENICE_INFOGRAPHIC_MODEL", "nano-banana-pro")
VENICE_VIDEO_MODEL = os.getenv("VENICE_VIDEO_MODEL", "wan-2.5-preview-image-to-video")
VENICE_TRANSCRIPTION_MODEL = os.getenv("VENICE_TRANSCRIPTION_MODEL", "nvidia/parakeet-tdt-0.6b-v3")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("generator")


# ── Database ───────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_type TEXT NOT NULL DEFAULT 'image',
            media_path TEXT NOT NULL,
            thumbnail_path TEXT,
            caption TEXT DEFAULT '',
            hashtags TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending',
            scheduled_at REAL NOT NULL,
            sort_order INTEGER DEFAULT 0,
            share_to_story INTEGER DEFAULT 1,
            story_status TEXT,
            instagram_media_id TEXT,
            error_message TEXT,
            retry_count INTEGER DEFAULT 0,
            created_at REAL NOT NULL,
            posted_at REAL
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS post_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER,
            action TEXT NOT NULL,
            details TEXT,
            timestamp REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            question_prompt TEXT DEFAULT '',
            enabled INTEGER DEFAULT 1,
            sort_order INTEGER DEFAULT 0,
            last_asked_at REAL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS character_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT DEFAULT ''
        );

        INSERT OR IGNORE INTO character_config (key, value) VALUES ('physical_description', '');
        INSERT OR IGNORE INTO character_config (key, value) VALUES ('style_descriptors', '');
        INSERT OR IGNORE INTO character_config (key, value) VALUES ('camera_preferences', '');
        INSERT OR IGNORE INTO character_config (key, value) VALUES ('brand_voice', '');
        INSERT OR IGNORE INTO character_config (key, value) VALUES ('default_hashtags', '');
        INSERT OR IGNORE INTO character_config (key, value) VALUES ('infographic_style', '');

        CREATE TABLE IF NOT EXISTS generation_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER,
            status TEXT NOT NULL DEFAULT 'questioning',
            question_text TEXT,
            response_text TEXT,
            response_type TEXT,
            voice_file_path TEXT,
            generated_caption TEXT,
            generated_hashtags TEXT,
            generated_media_type TEXT,
            generated_media_path TEXT,
            video_queue_id TEXT,
            post_id INTEGER,
            schedule_mode TEXT DEFAULT 'queue',
            scheduled_for REAL,
            telegram_msg_id INTEGER,
            error_message TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            FOREIGN KEY (topic_id) REFERENCES topics(id)
        );

        INSERT OR IGNORE INTO settings (key, value) VALUES ('generator_enabled', '0');
        INSERT OR IGNORE INTO settings (key, value) VALUES ('generator_question_interval_hours', '4');
        INSERT OR IGNORE INTO settings (key, value) VALUES ('generator_default_media_type', 'image');
        INSERT OR IGNORE INTO settings (key, value) VALUES ('generator_media_weights', '{"image":60,"infographic":25,"reel":15}');
        INSERT OR IGNORE INTO settings (key, value) VALUES ('generator_auto_post', '0');
        INSERT OR IGNORE INTO settings (key, value) VALUES ('generator_telegram_chat_id', '');
    """)
    # Migrate: add new columns if missing
    cols = [r[1] for r in db.execute("PRAGMA table_info(posts)").fetchall()]
    if "generation_session_id" not in cols:
        db.execute("ALTER TABLE posts ADD COLUMN generation_session_id INTEGER")
    db.commit()
    db.close()
    log.info("Database initialized at %s", DB_PATH)


def get_setting(key, default=""):
    db = get_db()
    row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    db.close()
    return row["value"] if row else default


def get_character_config():
    db = get_db()
    rows = db.execute("SELECT key, value FROM character_config").fetchall()
    db.close()
    return {r["key"]: r["value"] for r in rows}


def pick_media_type():
    """Weighted random selection of media type (image/infographic/reel)."""
    raw = get_setting("generator_media_weights", '{"image":60,"infographic":25,"reel":15}')
    try:
        weights = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return get_setting("generator_default_media_type", "image")
    types = list(weights.keys())
    vals = list(weights.values())
    return random.choices(types, weights=vals, k=1)[0]


def get_pending_session():
    """Get the current session awaiting user response."""
    db = get_db()
    row = db.execute(
        "SELECT * FROM generation_sessions WHERE status = 'awaiting_response' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    db.close()
    return row


def pick_next_topic():
    """Pick the next enabled topic using round-robin by last_asked_at."""
    db = get_db()
    topic = db.execute(
        """SELECT * FROM topics WHERE enabled = 1
           ORDER BY COALESCE(last_asked_at, 0) ASC, sort_order ASC
           LIMIT 1"""
    ).fetchone()
    db.close()
    return topic


def get_topic_by_name(name):
    db = get_db()
    topic = db.execute(
        "SELECT * FROM topics WHERE LOWER(name) = LOWER(?) AND enabled = 1", (name,)
    ).fetchone()
    db.close()
    return topic


# ── Venice API ─────────────────────────────────────────────────────────────────

async def venice_chat(messages, model=None, max_tokens=500, temperature=0.9):
    """Call Venice.ai chat completions (OpenAI-compatible)."""
    model = model or VENICE_CHAT_MODEL
    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{VENICE_BASE}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error("Venice chat error: %s", e)
        return None


async def venice_transcribe(audio_bytes, filename="audio.mp3"):
    """Transcribe audio via Venice AI."""
    headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            files = {"file": (filename, audio_bytes, "audio/mpeg")}
            data = {"model": VENICE_TRANSCRIPTION_MODEL}
            resp = await client.post(f"{VENICE_BASE}/audio/transcriptions", files=files, data=data, headers=headers)
            resp.raise_for_status()
            result = resp.json()
            return result.get("text", "").strip()
    except Exception as e:
        log.error("Venice transcription error: %s", e)
        return None


async def venice_generate_image(prompt, model=None, width=1024, height=1024):
    """Generate an image using Venice AI (text-to-image). Returns image bytes or None."""
    model = model or VENICE_IMAGE_MODEL
    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "width": width,
        "height": height,
        "format": "png",
        "cfg_scale": 7.5,
        "safe_mode": False,
        "hide_watermark": True,
        "variants": 1,
    }
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(f"{VENICE_BASE}/image/generate", json=payload, headers=headers)
                if resp.status_code == 503 and attempt < 2:
                    wait = (attempt + 1) * 10
                    log.warning("Venice image 503, retrying in %ds...", wait)
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code != 200:
                    log.error("Venice image API %d: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
                data = resp.json()
                img_b64 = data["images"][0]
                return base64.b64decode(img_b64)
        except Exception as e:
            log.error("Venice image generation error (attempt %d): %s", attempt + 1, e)
            if attempt < 2:
                await asyncio.sleep(5)
    return None


async def venice_edit_image(prompt, reference_image_path, model="flux-2-max-edit", aspect_ratio="1:1"):
    """Edit/transform an image using Venice AI edit endpoint. Returns image bytes or None."""
    headers = {"Authorization": f"Bearer {VENICE_API_KEY}"}
    ref_path = Path(reference_image_path)
    if not ref_path.exists():
        log.error("Reference image not found: %s", reference_image_path)
        return None
    img_bytes = ref_path.read_bytes()
    ext = ref_path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else "image/jpeg"

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                files = {"image": (ref_path.name, img_bytes, mime)}
                data = {"prompt": prompt, "modelId": model, "aspect_ratio": aspect_ratio}
                resp = await client.post(f"{VENICE_BASE}/image/edit", files=files, data=data, headers=headers)
                if resp.status_code == 503 and attempt < 2:
                    wait = (attempt + 1) * 10
                    log.warning("Venice edit 503, retrying in %ds...", wait)
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code != 200:
                    log.error("Venice edit API %d: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
                if "image" in resp.headers.get("content-type", ""):
                    return resp.content
                log.error("Venice edit returned non-image content-type: %s", resp.headers.get("content-type"))
                return None
        except Exception as e:
            log.error("Venice edit image error (attempt %d): %s", attempt + 1, e)
            if attempt < 2:
                await asyncio.sleep(5)
    return None


async def venice_video_queue(prompt, image_path, duration=5):
    """Queue a video generation job. Returns queue_id or None."""
    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json",
    }
    # Read image and encode as data URI
    img_bytes = Path(image_path).read_bytes()
    b64 = base64.b64encode(img_bytes).decode()
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else "image/jpeg"
    data_uri = f"data:{mime};base64,{b64}"

    payload = {
        "model": VENICE_VIDEO_MODEL,
        "prompt": prompt,
        "image": data_uri,
        "duration": duration,
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{VENICE_BASE}/video/queue", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data.get("id") or data.get("queue_id")
    except Exception as e:
        log.error("Venice video queue error: %s", e)
        return None


async def venice_video_poll(queue_id):
    """Poll for video completion. Returns video bytes when done, 'pending' if still processing, None on error."""
    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"id": queue_id}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{VENICE_BASE}/video/retrieve", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")
            if status in ("pending", "processing", "queued"):
                return "pending"
            if status == "completed" or data.get("video"):
                video_b64 = data.get("video", "")
                if video_b64:
                    # Cleanup
                    try:
                        async with httpx.AsyncClient(timeout=10) as c2:
                            await c2.post(f"{VENICE_BASE}/video/complete", json=payload, headers=headers)
                    except Exception:
                        pass
                    return base64.b64decode(video_b64)
            if status == "failed":
                log.error("Video generation failed: %s", data.get("error", "unknown"))
                return None
            return "pending"
    except Exception as e:
        log.error("Venice video poll error: %s", e)
        return None


# ── Audio conversion ───────────────────────────────────────────────────────────

def convert_ogg_to_mp3(ogg_path, mp3_path):
    """Convert OGG voice memo to MP3 using ffmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(ogg_path), "-acodec", "libmp3lame", "-q:a", "4", str(mp3_path)],
            capture_output=True, timeout=30, check=True,
        )
        return True
    except Exception as e:
        log.error("ffmpeg conversion error: %s", e)
        return False


# ── Content generation pipeline ───────────────────────────────────────────────

async def generate_question(topic):
    """Generate a personal, engaging question about a topic."""
    custom_prompt = topic["question_prompt"] if topic["question_prompt"] else ""
    system = "You are a creative content strategist. Generate a single, personal, engaging question for an Instagram content creator about the given topic. The question should prompt them to share a personal story, insight, or opinion. Keep it conversational and under 2 sentences. Only output the question, nothing else."
    if custom_prompt:
        system += f"\n\nAdditional guidance: {custom_prompt}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Topic: {topic['name']}\nDescription: {topic['description'] or topic['name']}"},
    ]
    return await venice_chat(messages, max_tokens=200)


async def generate_caption(topic_name, user_response, brand_voice=""):
    """Generate an Instagram caption from the user's response."""
    system = "You are an Instagram caption writer. Write a compelling, authentic Instagram caption based on the user's response to a question. Make it personal and engaging. Include a call-to-action or question for followers. Keep it under 300 words. Only output the caption text, no hashtags."
    if brand_voice:
        system += f"\n\nBrand voice guidelines: {brand_voice}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Topic: {topic_name}\nMy response: {user_response}"},
    ]
    return await venice_chat(messages, max_tokens=500)


async def generate_hashtags(topic_name, caption, default_hashtags=""):
    """Generate relevant hashtags for the caption."""
    system = "Generate 10-15 relevant Instagram hashtags for this post. Mix popular and niche hashtags. Output only the hashtags separated by spaces, starting with #."
    if default_hashtags:
        system += f"\n\nAlways include these hashtags: {default_hashtags}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Topic: {topic_name}\nCaption: {caption}"},
    ]
    return await venice_chat(messages, max_tokens=200, temperature=0.7)


async def generate_image_prompt(topic_name, user_response, char_config, edit_mode=False):
    """Generate an image generation prompt with character likeness."""
    physical = char_config.get("physical_description", "")
    style = char_config.get("style_descriptors", "")
    camera = char_config.get("camera_preferences", "")

    if edit_mode:
        system = """You are an AI image prompt engineer for an IMAGE EDITING model. The model receives a reference photo of a person and your prompt, then transforms the image accordingly.

IMPORTANT: The model already has the person's appearance from the reference photo — do NOT describe their physical features. Instead, focus entirely on TRANSFORMING the scene:

1. SETTING: Describe a vivid, specific new environment/location (not just a background swap — a completely different scene)
2. EXPRESSION & EMOTION: Specify a distinct facial expression and emotional state (laughing, contemplative, excited, serene, etc.)
3. POSE & BODY LANGUAGE: Describe what the person is doing — their pose, hand positions, body orientation
4. LIGHTING & ATMOSPHERE: Dramatic lighting, time of day, weather, mood
5. CLOTHING/STYLING: If relevant to the topic, suggest what they're wearing

Write the prompt as a direct transformation instruction. Be bold and creative — the more specific and different from a standard headshot, the better. Keep it under 120 words. Only output the prompt, nothing else."""
    else:
        system = """You are an AI image prompt engineer for a text-to-image model. Create a detailed, vivid image generation prompt for a lifestyle Instagram photo. The image should visually represent the topic and feeling of the user's response.

CRITICAL: If a person description is provided, you MUST include the FULL physical description verbatim at the start of your prompt — this is how the model maintains character likeness across images. Include every detail: hair, skin, body type, clothing style, distinguishing features.

Then add specific details about setting, lighting, composition, and mood that match the topic. Keep the total prompt under 150 words. Only output the prompt, nothing else."""

    context_parts = [f"Topic: {topic_name}", f"User's thoughts: {user_response[:300]}"]
    if not edit_mode and physical:
        context_parts.append(f"PERSON DESCRIPTION (include verbatim in prompt): {physical}")
    if style:
        context_parts.append(f"Style: {style}")
    if camera:
        context_parts.append(f"Camera: {camera}")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(context_parts)},
    ]
    return await venice_chat(messages, max_tokens=200, temperature=0.8)


async def generate_infographic_prompt(topic_name, user_response, char_config):
    """Generate a prompt for infographic-style image."""
    infographic_style = char_config.get("infographic_style", "")

    system = """You are an AI image prompt engineer. Create a prompt for an Instagram infographic image. The infographic should visualize key points from the user's response in a clean, shareable format. Include details about layout, colors, typography style, and visual elements. Keep it under 100 words. Only output the prompt, nothing else."""

    context_parts = [f"Topic: {topic_name}", f"Content: {user_response[:300]}"]
    if infographic_style:
        context_parts.append(f"Style preferences: {infographic_style}")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(context_parts)},
    ]
    return await venice_chat(messages, max_tokens=200, temperature=0.7)


async def generate_content(session_id):
    """Orchestrate the full content generation pipeline for a session."""
    db = get_db()
    session = db.execute("SELECT * FROM generation_sessions WHERE id = ?", (session_id,)).fetchone()
    if not session:
        db.close()
        return None

    topic = db.execute("SELECT * FROM topics WHERE id = ?", (session["topic_id"],)).fetchone()
    topic_name = topic["name"] if topic else "General"
    user_response = session["response_text"] or ""
    media_type = session["generated_media_type"] or get_setting("generator_default_media_type", "image")

    db.execute("UPDATE generation_sessions SET status = 'generating', updated_at = ? WHERE id = ?",
               (time.time(), session_id))
    db.commit()
    db.close()

    char_config = get_character_config()

    # Generate caption
    brand_voice = char_config.get("brand_voice", "")
    caption = await generate_caption(topic_name, user_response, brand_voice)
    if not caption:
        _fail_session(session_id, "Failed to generate caption")
        return None

    # Generate hashtags
    default_tags = char_config.get("default_hashtags", "")
    hashtags = await generate_hashtags(topic_name, caption, default_tags)

    # Resolve reference image
    ref_image = char_config.get("reference_image", "")
    ref_image_path = str(UPLOADS_DIR / ref_image) if ref_image and (UPLOADS_DIR / ref_image).exists() else None

    # Generate media
    media_path = None
    if media_type == "image":
        img_prompt = await generate_image_prompt(topic_name, user_response, char_config, edit_mode=bool(ref_image_path))
        if img_prompt:
            log.info("Image prompt: %s", img_prompt[:100])
            if ref_image_path:
                log.info("Using edit API with reference image")
                img_bytes = await venice_edit_image(img_prompt, ref_image_path)
            else:
                img_bytes = await venice_generate_image(img_prompt, model=VENICE_IMAGE_MODEL)
            if img_bytes:
                filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
                (UPLOADS_DIR / filename).write_bytes(img_bytes)
                media_path = filename
                log.info("Image saved: %s", filename)

    elif media_type == "infographic":
        info_prompt = await generate_infographic_prompt(topic_name, user_response, char_config)
        if info_prompt:
            log.info("Infographic prompt: %s", info_prompt[:100])
            img_bytes = await venice_generate_image(info_prompt, model=VENICE_INFOGRAPHIC_MODEL, width=1080, height=1350)
            if img_bytes:
                filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
                (UPLOADS_DIR / filename).write_bytes(img_bytes)
                media_path = filename
                log.info("Infographic saved: %s", filename)

    elif media_type == "reel":
        # Generate still image first, then queue for video
        img_prompt = await generate_image_prompt(topic_name, user_response, char_config, edit_mode=bool(ref_image_path))
        if img_prompt:
            if ref_image_path:
                log.info("Using edit API with reference image for reel still")
                img_bytes = await venice_edit_image(img_prompt, ref_image_path)
            else:
                img_bytes = await venice_generate_image(img_prompt, model=VENICE_IMAGE_MODEL)
            if img_bytes:
                still_filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_still.png"
                still_path = UPLOADS_DIR / still_filename
                still_path.write_bytes(img_bytes)
                log.info("Still image for reel saved: %s", still_filename)

                # Queue video generation
                queue_id = await venice_video_queue(
                    f"Smooth cinematic motion, {img_prompt[:200]}",
                    str(still_path),
                )
                if queue_id:
                    db = get_db()
                    db.execute(
                        """UPDATE generation_sessions SET
                           generated_caption = ?, generated_hashtags = ?,
                           generated_media_type = 'reel', video_queue_id = ?,
                           status = 'generating', updated_at = ?
                           WHERE id = ?""",
                        (caption, hashtags or "", queue_id, time.time(), session_id),
                    )
                    db.commit()
                    db.close()
                    log.info("Video queued: %s", queue_id)
                    return {"status": "video_queued", "queue_id": queue_id}
                else:
                    # Fall back to image if video queue fails
                    media_path = still_filename
                    media_type = "image"
                    log.warning("Video queue failed, falling back to image")

    if not media_path:
        _fail_session(session_id, "Failed to generate media")
        return None

    # Update session with generated content
    db = get_db()
    db.execute(
        """UPDATE generation_sessions SET
           generated_caption = ?, generated_hashtags = ?,
           generated_media_type = ?, generated_media_path = ?,
           status = 'content_ready', updated_at = ?
           WHERE id = ?""",
        (caption, hashtags or "", media_type, media_path, time.time(), session_id),
    )
    db.commit()
    db.close()

    log.info("Content generated for session #%d: %s", session_id, media_type)
    return {
        "status": "content_ready",
        "caption": caption,
        "hashtags": hashtags,
        "media_type": media_type,
        "media_path": media_path,
    }


def _fail_session(session_id, error_msg):
    db = get_db()
    db.execute(
        "UPDATE generation_sessions SET status = 'failed', error_message = ?, updated_at = ? WHERE id = ?",
        (error_msg, time.time(), session_id),
    )
    db.commit()
    db.close()
    log.error("Session #%d failed: %s", session_id, error_msg)


def insert_into_posts(session_id, mode="queue"):
    """Insert a content_ready session into the posts table."""
    db = get_db()
    sess = db.execute("SELECT * FROM generation_sessions WHERE id = ? AND status = 'content_ready'", (session_id,)).fetchone()
    if not sess:
        db.close()
        return None

    media_type = sess["generated_media_type"] or "image"
    post_type = "reel" if media_type == "reel" else "image"

    if mode == "now":
        scheduled_at = time.time() - 1
    elif mode == "queue":
        last_pending = db.execute("SELECT MAX(scheduled_at) as last_ts FROM posts WHERE status = 'pending'").fetchone()
        last_ts = last_pending["last_ts"] if last_pending and last_pending["last_ts"] else time.time()
        scheduled_at = max(last_ts + 3600, time.time())
    else:
        scheduled_at = time.time()

    max_order = db.execute("SELECT COALESCE(MAX(sort_order), 0) FROM posts WHERE status = 'pending'").fetchone()[0]
    db.execute(
        """INSERT INTO posts (post_type, media_path, caption, hashtags, status, scheduled_at,
           sort_order, share_to_story, generation_session_id, created_at)
           VALUES (?, ?, ?, ?, 'pending', ?, ?, 1, ?, ?)""",
        (post_type, sess["generated_media_path"], sess["generated_caption"] or "",
         sess["generated_hashtags"] or "", scheduled_at, max_order + 1, session_id, time.time()),
    )
    post_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    db.execute(
        "UPDATE generation_sessions SET status = 'inserted', post_id = ?, updated_at = ? WHERE id = ?",
        (post_id, time.time(), session_id),
    )
    db.execute(
        "INSERT INTO post_log (post_id, action, details, timestamp) VALUES (?, 'queued', 'Added from generator bot', ?)",
        (post_id, time.time()),
    )
    db.commit()
    db.close()
    log.info("Post #%d created from session #%d (%s)", post_id, session_id, mode)
    return post_id


# ── Authorization middleware ──────────────────────────────────────────────────

def is_authorized(user_id):
    if not ALLOWED_USERS:
        return True
    return user_id in ALLOWED_USERS


# ── Telegram command handlers ─────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        return
    await update.message.reply_text(
        "IG Content Generator Bot\n\n"
        "Commands:\n"
        "/generate [topic] — Get a question (random or specific topic)\n"
        "/skip — Cancel current question\n"
        "/status — Show generator status\n"
        "/topics — List available topics\n"
        "/type image|reel|infographic — Set media type for next generation\n"
    )


async def cmd_generate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        return

    # Check for existing pending session
    pending = get_pending_session()
    if pending:
        await update.message.reply_text(
            "You already have a pending question. Reply to it or use /skip to cancel."
        )
        return

    # Pick topic
    topic_name = " ".join(context.args) if context.args else None
    if topic_name:
        topic = get_topic_by_name(topic_name)
        if not topic:
            await update.message.reply_text(f"Topic '{topic_name}' not found. Use /topics to see available topics.")
            return
    else:
        topic = pick_next_topic()
        if not topic:
            await update.message.reply_text("No topics configured. Add topics via the dashboard.")
            return

    await send_question(update.effective_chat.id, topic, context)


async def send_question(chat_id, topic, context):
    """Generate and send a question for the given topic."""
    question = await generate_question(topic)
    if not question:
        await context.bot.send_message(chat_id, "Failed to generate a question. Try again later.")
        return

    # Create session
    now = time.time()
    db = get_db()
    db.execute(
        """INSERT INTO generation_sessions
           (topic_id, status, question_text, generated_media_type, created_at, updated_at)
           VALUES (?, 'awaiting_response', ?, ?, ?, ?)""",
        (topic["id"], question, pick_media_type(), now, now),
    )
    session_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    # Update topic last_asked_at
    db.execute("UPDATE topics SET last_asked_at = ? WHERE id = ?", (now, topic["id"]))
    db.commit()
    db.close()

    msg = await context.bot.send_message(
        chat_id,
        f"*{topic['name']}*\n\n{question}\n\n_Reply with text or a voice memo._",
        parse_mode="Markdown",
    )

    # Store telegram message id
    db = get_db()
    db.execute("UPDATE generation_sessions SET telegram_msg_id = ? WHERE id = ?", (msg.message_id, session_id))
    db.commit()
    db.close()

    log.info("Question sent for topic '%s' (session #%d)", topic["name"], session_id)


async def cmd_skip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        return

    pending = get_pending_session()
    if not pending:
        await update.message.reply_text("No pending question to skip.")
        return

    db = get_db()
    db.execute(
        "UPDATE generation_sessions SET status = 'cancelled', updated_at = ? WHERE id = ?",
        (time.time(), pending["id"]),
    )
    db.commit()
    db.close()
    await update.message.reply_text("Question skipped.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        return

    db = get_db()
    pending_sessions = db.execute("SELECT COUNT(*) FROM generation_sessions WHERE status = 'awaiting_response'").fetchone()[0]
    generating = db.execute("SELECT COUNT(*) FROM generation_sessions WHERE status = 'generating'").fetchone()[0]
    ready = db.execute("SELECT COUNT(*) FROM generation_sessions WHERE status = 'content_ready'").fetchone()[0]
    inserted = db.execute("SELECT COUNT(*) FROM generation_sessions WHERE status = 'inserted'").fetchone()[0]
    total_topics = db.execute("SELECT COUNT(*) FROM topics WHERE enabled = 1").fetchone()[0]
    pending_posts = db.execute("SELECT COUNT(*) FROM posts WHERE status = 'pending'").fetchone()[0]
    db.close()

    enabled = get_setting("generator_enabled", "0")
    interval = get_setting("generator_question_interval_hours", "4")
    raw_weights = get_setting("generator_media_weights", '{"image":60,"infographic":25,"reel":15}')
    try:
        weights = json.loads(raw_weights)
        total = sum(weights.values())
        weights_str = ", ".join(f"{k} {int(v/total*100)}%" for k, v in weights.items())
    except Exception:
        weights_str = "image 100%"

    await update.message.reply_text(
        f"Generator: {'Enabled' if enabled == '1' else 'Disabled'}\n"
        f"Question interval: {interval}h\n"
        f"Media mix: {weights_str}\n"
        f"Active topics: {total_topics}\n\n"
        f"Awaiting response: {pending_sessions}\n"
        f"Generating: {generating}\n"
        f"Content ready: {ready}\n"
        f"Inserted to queue: {inserted}\n"
        f"Pending posts: {pending_posts}"
    )


async def cmd_topics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        return

    db = get_db()
    topics = db.execute("SELECT * FROM topics ORDER BY sort_order ASC").fetchall()
    db.close()

    if not topics:
        await update.message.reply_text("No topics configured. Add them via the dashboard.")
        return

    lines = []
    for t in topics:
        status = "ON" if t["enabled"] else "OFF"
        lines.append(f"{'>' if t['enabled'] else ' '} {t['name']} [{status}]")

    await update.message.reply_text("Topics:\n" + "\n".join(lines))


async def cmd_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.effective_user.id):
        return

    if not context.args:
        current = get_setting("generator_default_media_type", "image")
        await update.message.reply_text(f"Current default media type: {current}\nUsage: /type image|reel|infographic")
        return

    media_type = context.args[0].lower()
    if media_type not in ("image", "reel", "infographic"):
        await update.message.reply_text("Invalid type. Choose: image, reel, infographic")
        return

    # Update the pending session's media type if one exists
    pending = get_pending_session()
    if pending:
        db = get_db()
        db.execute("UPDATE generation_sessions SET generated_media_type = ?, updated_at = ? WHERE id = ?",
                   (media_type, time.time(), pending["id"]))
        db.commit()
        db.close()
        await update.message.reply_text(f"Media type for current session set to: {media_type}")
    else:
        # Update the default
        db = get_db()
        db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('generator_default_media_type', ?)", (media_type,))
        db.commit()
        db.close()
        await update.message.reply_text(f"Default media type set to: {media_type}")


# ── Message handlers ──────────────────────────────────────────────────────────

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text replies to questions."""
    if not is_authorized(update.effective_user.id):
        return

    pending = get_pending_session()
    if not pending:
        return  # No pending session, ignore

    user_text = update.message.text.strip()
    if not user_text:
        return

    session_id = pending["id"]
    db = get_db()
    db.execute(
        "UPDATE generation_sessions SET response_text = ?, response_type = 'text', status = 'generating', updated_at = ? WHERE id = ?",
        (user_text, time.time(), session_id),
    )
    db.commit()
    db.close()

    await update.message.reply_text("Got it! Generating your content...")
    log.info("Text response received for session #%d", session_id)

    # Run generation
    result = await generate_content(session_id)
    if result and result.get("status") == "content_ready":
        await send_preview(update.effective_chat.id, session_id, context)
    elif result and result.get("status") == "video_queued":
        await update.message.reply_text("Video generation queued. I'll send a preview when it's ready.")
    else:
        await update.message.reply_text("Content generation failed. Check the dashboard for details.")


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice memo replies."""
    if not is_authorized(update.effective_user.id):
        return

    pending = get_pending_session()
    if not pending:
        return

    session_id = pending["id"]

    db = get_db()
    db.execute(
        "UPDATE generation_sessions SET response_type = 'voice', status = 'transcribing', updated_at = ? WHERE id = ?",
        (time.time(), session_id),
    )
    db.commit()
    db.close()

    await update.message.reply_text("Transcribing your voice memo...")

    # Download voice file
    voice = update.message.voice
    tg_file = await voice.get_file()
    ogg_path = UPLOADS_DIR / f"voice_{session_id}.ogg"
    mp3_path = UPLOADS_DIR / f"voice_{session_id}.mp3"

    await tg_file.download_to_drive(str(ogg_path))

    # Convert OGG to MP3
    if not convert_ogg_to_mp3(ogg_path, mp3_path):
        _fail_session(session_id, "Audio conversion failed")
        await update.message.reply_text("Failed to process voice memo. Try sending text instead.")
        # Cleanup
        ogg_path.unlink(missing_ok=True)
        return

    # Transcribe
    mp3_bytes = mp3_path.read_bytes()
    transcript = await venice_transcribe(mp3_bytes, f"voice_{session_id}.mp3")

    # Cleanup audio files
    ogg_path.unlink(missing_ok=True)
    mp3_path.unlink(missing_ok=True)

    if not transcript:
        _fail_session(session_id, "Transcription failed")
        await update.message.reply_text("Transcription failed. Try sending text instead.")
        return

    log.info("Transcribed voice for session #%d: %s", session_id, transcript[:100])
    await update.message.reply_text(f"Transcribed:\n_{transcript[:500]}_\n\nGenerating content...", parse_mode="Markdown")

    db = get_db()
    db.execute(
        "UPDATE generation_sessions SET response_text = ?, voice_file_path = ?, status = 'generating', updated_at = ? WHERE id = ?",
        (transcript, f"voice_{session_id}.ogg", time.time(), session_id),
    )
    db.commit()
    db.close()

    # Run generation
    result = await generate_content(session_id)
    if result and result.get("status") == "content_ready":
        await send_preview(update.effective_chat.id, session_id, context)
    elif result and result.get("status") == "video_queued":
        await update.message.reply_text("Video generation queued. I'll send a preview when it's ready.")
    else:
        await update.message.reply_text("Content generation failed. Check the dashboard for details.")


# ── Preview + inline keyboards ────────────────────────────────────────────────

async def send_preview(chat_id, session_id, context):
    """Send a content preview with inline action buttons."""
    db = get_db()
    session = db.execute("SELECT * FROM generation_sessions WHERE id = ?", (session_id,)).fetchone()
    db.close()
    if not session:
        return

    caption_preview = (session["generated_caption"] or "")[:500]
    hashtags_preview = (session["generated_hashtags"] or "")[:200]
    media_type = session["generated_media_type"] or "image"
    media_path = session["generated_media_path"]

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Queue", callback_data=f"approve_queue:{session_id}"),
            InlineKeyboardButton("Post Now", callback_data=f"approve_now:{session_id}"),
        ],
        [
            InlineKeyboardButton("Redo All", callback_data=f"redo_all:{session_id}"),
            InlineKeyboardButton("Redo Caption", callback_data=f"redo_caption:{session_id}"),
            InlineKeyboardButton("Redo Image", callback_data=f"redo_image:{session_id}"),
        ],
        [
            InlineKeyboardButton("Cancel", callback_data=f"cancel:{session_id}"),
        ],
    ])

    text = f"*Preview* ({media_type})\n\n{caption_preview}\n\n{hashtags_preview}"

    if media_path and (UPLOADS_DIR / media_path).exists():
        full_path = UPLOADS_DIR / media_path
        if media_type == "reel" and media_path.endswith(".mp4"):
            await context.bot.send_video(
                chat_id, video=open(str(full_path), "rb"),
                caption=text, parse_mode="Markdown", reply_markup=keyboard,
            )
        else:
            await context.bot.send_photo(
                chat_id, photo=open(str(full_path), "rb"),
                caption=text, parse_mode="Markdown", reply_markup=keyboard,
            )
    else:
        await context.bot.send_message(
            chat_id, text=text + "\n\n(media file missing)",
            parse_mode="Markdown", reply_markup=keyboard,
        )


# ── Callback query handlers ──────────────────────────────────────────────────

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not is_authorized(query.from_user.id):
        await query.answer("Unauthorized")
        return

    await query.answer()
    data = query.data
    action, session_id_str = data.split(":", 1)
    session_id = int(session_id_str)

    if action == "approve_queue":
        post_id = insert_into_posts(session_id, mode="queue")
        if post_id:
            await query.edit_message_caption(
                caption=f"Queued as post #{post_id}",
            )
        else:
            await query.edit_message_caption(caption="Failed to queue. Check dashboard.")

    elif action == "approve_now":
        post_id = insert_into_posts(session_id, mode="now")
        if post_id:
            await query.edit_message_caption(
                caption=f"Post #{post_id} scheduled for immediate publishing!",
            )
        else:
            await query.edit_message_caption(caption="Failed to queue. Check dashboard.")

    elif action == "redo_all":
        db = get_db()
        sess = db.execute("SELECT * FROM generation_sessions WHERE id = ?", (session_id,)).fetchone()
        if sess and sess["generated_media_path"]:
            # Delete old media
            old_path = UPLOADS_DIR / sess["generated_media_path"]
            old_path.unlink(missing_ok=True)
        db.execute(
            """UPDATE generation_sessions SET status = 'generating',
               generated_caption = NULL, generated_hashtags = NULL,
               generated_media_path = NULL, updated_at = ?
               WHERE id = ?""",
            (time.time(), session_id),
        )
        db.commit()
        db.close()

        await query.edit_message_caption(caption="Regenerating everything...")
        result = await generate_content(session_id)
        if result and result.get("status") == "content_ready":
            await send_preview(query.message.chat.id, session_id, context)
        elif result and result.get("status") == "video_queued":
            await context.bot.send_message(query.message.chat.id, "Video generation queued. I'll send a preview when it's ready.")
        else:
            await context.bot.send_message(query.message.chat.id, "Regeneration failed.")

    elif action == "redo_caption":
        db = get_db()
        sess = db.execute("SELECT * FROM generation_sessions WHERE id = ?", (session_id,)).fetchone()
        if not sess:
            db.close()
            return
        topic = db.execute("SELECT * FROM topics WHERE id = ?", (sess["topic_id"],)).fetchone()
        db.close()

        topic_name = topic["name"] if topic else "General"
        char_config = get_character_config()
        brand_voice = char_config.get("brand_voice", "")
        new_caption = await generate_caption(topic_name, sess["response_text"] or "", brand_voice)
        default_tags = char_config.get("default_hashtags", "")
        new_hashtags = await generate_hashtags(topic_name, new_caption or "", default_tags)

        if new_caption:
            db = get_db()
            db.execute(
                "UPDATE generation_sessions SET generated_caption = ?, generated_hashtags = ?, updated_at = ? WHERE id = ?",
                (new_caption, new_hashtags or "", time.time(), session_id),
            )
            db.commit()
            db.close()
            await send_preview(query.message.chat.id, session_id, context)
        else:
            await context.bot.send_message(query.message.chat.id, "Caption regeneration failed.")

    elif action == "redo_image":
        db = get_db()
        sess = db.execute("SELECT * FROM generation_sessions WHERE id = ?", (session_id,)).fetchone()
        if not sess:
            db.close()
            return
        topic = db.execute("SELECT * FROM topics WHERE id = ?", (sess["topic_id"],)).fetchone()
        # Delete old media
        if sess["generated_media_path"]:
            old_path = UPLOADS_DIR / sess["generated_media_path"]
            old_path.unlink(missing_ok=True)
        db.close()

        topic_name = topic["name"] if topic else "General"
        char_config = get_character_config()
        media_type = sess["generated_media_type"] or "image"
        ref_image = char_config.get("reference_image", "")
        ref_path = str(UPLOADS_DIR / ref_image) if ref_image and (UPLOADS_DIR / ref_image).exists() else None

        await query.edit_message_caption(caption="Regenerating image...")

        if media_type == "infographic":
            img_prompt = await generate_infographic_prompt(topic_name, sess["response_text"] or "", char_config)
            img_bytes = await venice_generate_image(img_prompt, model=VENICE_INFOGRAPHIC_MODEL, width=1080, height=1350)
        elif ref_path:
            img_prompt = await generate_image_prompt(topic_name, sess["response_text"] or "", char_config, edit_mode=True)
            img_bytes = await venice_edit_image(img_prompt, ref_path) if img_prompt else None
        else:
            img_prompt = await generate_image_prompt(topic_name, sess["response_text"] or "", char_config)
            img_bytes = await venice_generate_image(img_prompt, model=VENICE_IMAGE_MODEL) if img_prompt else None

        if img_prompt and img_bytes:
            if img_bytes:
                filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
                (UPLOADS_DIR / filename).write_bytes(img_bytes)

                db = get_db()
                db.execute(
                    "UPDATE generation_sessions SET generated_media_path = ?, updated_at = ? WHERE id = ?",
                    (filename, time.time(), session_id),
                )
                db.commit()
                db.close()
                await send_preview(query.message.chat.id, session_id, context)
                return

        await context.bot.send_message(query.message.chat.id, "Image regeneration failed.")

    elif action == "cancel":
        db = get_db()
        db.execute(
            "UPDATE generation_sessions SET status = 'cancelled', updated_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        db.commit()
        db.close()
        await query.edit_message_caption(caption="Session cancelled.")


# ── Background tasks ──────────────────────────────────────────────────────────

async def question_scheduler(context: ContextTypes.DEFAULT_TYPE):
    """Periodically send topic questions if generator is enabled."""
    enabled = get_setting("generator_enabled", "0")
    if enabled != "1":
        return

    chat_id = get_setting("generator_telegram_chat_id", "") or CHAT_ID
    if not chat_id:
        return

    # Don't send if there's already a pending session
    pending = get_pending_session()
    if pending:
        return

    # Check interval
    interval_hours = float(get_setting("generator_question_interval_hours", "4"))
    interval_seconds = interval_hours * 3600

    db = get_db()
    last_session = db.execute(
        "SELECT created_at FROM generation_sessions ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    db.close()

    if last_session:
        elapsed = time.time() - last_session["created_at"]
        if elapsed < interval_seconds:
            return

    # Pick topic and send question
    topic = pick_next_topic()
    if not topic:
        return

    log.info("Scheduler: sending question for topic '%s'", topic["name"])
    await send_question(int(chat_id), topic, context)


async def video_poller(context: ContextTypes.DEFAULT_TYPE):
    """Poll for pending video generation completions."""
    db = get_db()
    sessions = db.execute(
        "SELECT * FROM generation_sessions WHERE status = 'generating' AND video_queue_id IS NOT NULL"
    ).fetchall()
    db.close()

    for sess in sessions:
        queue_id = sess["video_queue_id"]
        result = await venice_video_poll(queue_id)

        if result == "pending":
            continue
        elif result is None:
            _fail_session(sess["id"], "Video generation failed")
            chat_id = get_setting("generator_telegram_chat_id", "") or CHAT_ID
            if chat_id:
                await context.bot.send_message(int(chat_id), f"Video generation failed for session #{sess['id']}.")
        elif isinstance(result, bytes):
            # Video is ready
            filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
            (UPLOADS_DIR / filename).write_bytes(result)
            log.info("Video saved: %s (session #%d)", filename, sess["id"])

            db = get_db()
            db.execute(
                """UPDATE generation_sessions SET
                   generated_media_path = ?, status = 'content_ready', updated_at = ?
                   WHERE id = ?""",
                (filename, time.time(), sess["id"]),
            )
            db.commit()
            db.close()

            chat_id = get_setting("generator_telegram_chat_id", "") or CHAT_ID
            if chat_id:
                await send_preview(int(chat_id), sess["id"], context)


async def process_dashboard_generations(context: ContextTypes.DEFAULT_TYPE):
    """Pick up sessions created from the dashboard (status='pending_generation') and process them."""
    db = get_db()
    sessions = db.execute(
        "SELECT * FROM generation_sessions WHERE status = 'pending_generation' ORDER BY created_at ASC LIMIT 1"
    ).fetchall()
    db.close()

    for sess in sessions:
        session_id = sess["id"]
        log.info("Processing dashboard generation session #%d", session_id)

        # Set to generating
        db = get_db()
        db.execute("UPDATE generation_sessions SET status = 'generating', updated_at = ? WHERE id = ?",
                   (time.time(), session_id))
        db.commit()
        db.close()

        result = await generate_content(session_id)

        if result and result.get("status") == "content_ready":
            # Auto-queue if requested
            db = get_db()
            sess_updated = db.execute("SELECT * FROM generation_sessions WHERE id = ?", (session_id,)).fetchone()
            db.close()
            if sess_updated and sess_updated["schedule_mode"] == "auto_queue":
                post_id = insert_into_posts(session_id, mode="queue")
                log.info("Dashboard session #%d auto-queued as post #%s", session_id, post_id)

            # Notify via Telegram if chat_id is set
            chat_id = get_setting("generator_telegram_chat_id", "") or CHAT_ID
            if chat_id:
                try:
                    await send_preview(int(chat_id), session_id, context)
                except Exception as e:
                    log.warning("Could not send Telegram preview for dashboard session: %s", e)

        elif result and result.get("status") == "video_queued":
            chat_id = get_setting("generator_telegram_chat_id", "") or CHAT_ID
            if chat_id:
                try:
                    await context.bot.send_message(int(chat_id), f"Video generation queued for session #{session_id}.")
                except Exception:
                    pass
        else:
            log.error("Dashboard generation failed for session #%d", session_id)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("Instagram Content Generator starting")

    if not BOT_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN must be set in .env")
        sys.exit(1)
    if not VENICE_API_KEY:
        log.error("VENICE_API_KEY must be set in .env")
        sys.exit(1)

    init_db()

    # Build the Telegram bot application
    app = Application.builder().token(BOT_TOKEN).build()

    # Command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("generate", cmd_generate))
    app.add_handler(CommandHandler("skip", cmd_skip))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("topics", cmd_topics))
    app.add_handler(CommandHandler("type", cmd_type))

    # Message handlers (voice before text so voice memos match first)
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    # Callback query handler for inline keyboards
    app.add_handler(CallbackQueryHandler(handle_callback))

    # Background tasks
    job_queue = app.job_queue
    job_queue.run_repeating(question_scheduler, interval=300, first=30)  # Check every 5 minutes
    job_queue.run_repeating(video_poller, interval=30, first=60)  # Poll every 30 seconds
    job_queue.run_repeating(process_dashboard_generations, interval=15, first=10)  # Check every 15 seconds

    log.info("Bot starting polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
