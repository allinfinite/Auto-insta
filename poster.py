"""
Instagram Autoposter — Background scheduler daemon.
Checks the queue every CHECK_INTERVAL seconds and publishes due posts.
Usage: python3 poster.py
"""

import json
import logging
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from instagrapi import Client
from instagrapi.exceptions import (
    LoginRequired,
    ChallengeRequired,
    TwoFactorRequired,
    ClientError,
)
from instagrapi.types import StoryMedia

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "instagram_autoposter.db"
UPLOADS_DIR = BASE_DIR / "uploads"
SESSION_FILE = BASE_DIR / "ig_session.json"
LOG_FILE = BASE_DIR / "poster.log"

UPLOADS_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
IG_USERNAME = os.getenv("INSTAGRAM_USERNAME", "")
IG_PASSWORD = os.getenv("INSTAGRAM_PASSWORD", "")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("poster")


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

        CREATE TABLE IF NOT EXISTS story_reshares (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instagram_media_id TEXT NOT NULL,
            shared_at REAL NOT NULL
        );

        INSERT OR IGNORE INTO settings (key, value) VALUES ('default_hashtags', '');
        INSERT OR IGNORE INTO settings (key, value) VALUES ('post_delay_seconds', '0');
        INSERT OR IGNORE INTO settings (key, value) VALUES ('auto_share_story', '1');
        INSERT OR IGNORE INTO settings (key, value) VALUES ('story_reshare_enabled', '0');
        INSERT OR IGNORE INTO settings (key, value) VALUES ('story_reshare_count', '5');

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
        INSERT OR IGNORE INTO settings (key, value) VALUES ('generator_auto_post', '0');
        INSERT OR IGNORE INTO settings (key, value) VALUES ('generator_telegram_chat_id', '');
    """)
    # Migrate: add new columns if missing
    cols = [r[1] for r in db.execute("PRAGMA table_info(posts)").fetchall()]
    if "share_to_story" not in cols:
        db.execute("ALTER TABLE posts ADD COLUMN share_to_story INTEGER DEFAULT 1")
    if "story_status" not in cols:
        db.execute("ALTER TABLE posts ADD COLUMN story_status TEXT")
    if "generation_session_id" not in cols:
        db.execute("ALTER TABLE posts ADD COLUMN generation_session_id INTEGER")
    db.commit()
    db.close()
    log.info("Database initialized at %s", DB_PATH)


def get_setting(db, key, default=""):
    row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def log_action(db, post_id, action, details=""):
    db.execute(
        "INSERT INTO post_log (post_id, action, details, timestamp) VALUES (?, ?, ?, ?)",
        (post_id, action, details, time.time()),
    )
    db.commit()


# ── Instagram client ──────────────────────────────────────────────────────────

DEVICE_SETTINGS = {
    "app_version": "326.0.0.39.96",
    "android_version": 33,
    "android_release": "13.0",
    "dpi": "420dpi",
    "resolution": "1080x2400",
    "manufacturer": "Google",
    "device": "cheetah",
    "model": "Pixel 7 Pro",
    "cpu": "tensor",
    "version_code": "567891234",
}
USER_AGENT = "Instagram 326.0.0.39.96 Android (33/13.0; 420dpi; 1080x2400; Google; Pixel 7 Pro; cheetah; tensor; en_US; 567891234)"

cl = Client()


def apply_device():
    """Apply device settings and user agent. Call after any load_settings."""
    cl.set_device(DEVICE_SETTINGS)
    cl.set_user_agent(USER_AGENT)
    cl.delay_range = [1, 3]


def challenge_code_handler(username, choice):
    """Prompt for Instagram challenge verification code (email/SMS)."""
    log.info("Instagram challenge triggered for %s (method: %s)", username, choice)
    code = input(f"Enter challenge code for {username} ({choice}): ").strip()
    return code


apply_device()
cl.challenge_code_handler = challenge_code_handler


def _try_login(verification_code=""):
    """Attempt login, handling challenges inline. Returns True on success."""
    apply_device()
    try:
        cl.login(IG_USERNAME, IG_PASSWORD, verification_code=verification_code)
        cl.dump_settings(str(SESSION_FILE))
        return True
    except ChallengeRequired:
        log.info("Challenge required, resolving...")
        try:
            cl.challenge_resolve(cl.last_json)
            cl.dump_settings(str(SESSION_FILE))
            return True
        except Exception as e2:
            log.error("Challenge resolution failed: %s", e2)
            return False


def warmup_session():
    """Do normal browsing activity after login so Instagram trusts the session."""
    log.info("Warming up session...")
    try:
        cl.account_info()
        time.sleep(3)
    except Exception:
        pass
    try:
        cl.get_timeline_feed()
        time.sleep(3)
    except Exception:
        pass
    log.info("Warmup complete")


def ig_login():
    # Try saved session first
    if SESSION_FILE.exists():
        try:
            cl.load_settings(str(SESSION_FILE))
            apply_device()
            cl.login(IG_USERNAME, IG_PASSWORD)
            cl.get_timeline_feed()
            log.info("Logged in via saved session")
            return True
        except Exception as e:
            log.warning("Session login failed (%s), doing fresh login", e)

    # Fresh login
    try:
        if _try_login():
            log.info("Fresh login successful, session saved")
            warmup_session()
            return True
    except TwoFactorRequired:
        log.info("Two-factor authentication required")
        code = input("Enter 2FA code: ").strip()
        if not code:
            log.error("No 2FA code provided")
            return False
        try:
            if _try_login(verification_code=code):
                log.info("2FA login successful, session saved")
                warmup_session()
                return True
        except Exception as e:
            log.error("2FA login failed: %s", e)
            return False
    except Exception as e:
        log.error("Login failed: %s", e)
        return False

    log.error("Login failed")
    return False


def ig_relogin():
    log.info("Attempting re-login...")
    try:
        cl.login(IG_USERNAME, IG_PASSWORD, relogin=True)
        cl.dump_settings(str(SESSION_FILE))
        log.info("Re-login successful")
        return True
    except ChallengeRequired:
        try:
            cl.challenge_resolve(cl.last_json)
            cl.dump_settings(str(SESSION_FILE))
            log.info("Re-login successful after challenge")
            return True
        except Exception as e:
            log.error("Re-login challenge failed: %s", e)
            return False
    except Exception as e:
        log.error("Re-login failed: %s", e)
        return False


# ── Post publishing ────────────────────────────────────────────────────────────

def build_caption(caption, hashtags):
    parts = []
    if caption:
        parts.append(caption)
    if hashtags:
        parts.append(hashtags)
    return "\n\n".join(parts)


def share_post_to_story(media_pk, media_path):
    """Share a published feed post to story as an embedded media sticker."""
    try:
        media_pk_int = int(media_pk)
    except (ValueError, TypeError):
        log.warning("Cannot share to story: invalid media_pk %s", media_pk)
        return False

    log.info("Sharing post (media_pk: %s) to story...", media_pk)
    try:
        story_media = StoryMedia(media_pk=media_pk_int)
        path = UPLOADS_DIR / media_path
        if not path.exists():
            log.warning("Media file missing for story share: %s", path)
            return False
        ext = path.suffix.lower()
        if ext in (".mp4", ".mov", ".avi"):
            cl.video_upload_to_story(str(path), medias=[story_media])
        else:
            cl.photo_upload_to_story(str(path), medias=[story_media])
        log.info("Post shared to story successfully")
        return True
    except Exception as e:
        log.error("Failed to share post to story: %s", e)
        return False


def do_post(row, _resolved_challenge=False):
    media_path = UPLOADS_DIR / row["media_path"]
    if not media_path.exists():
        return False, f"Media file not found: {media_path}"

    caption = build_caption(row["caption"], row["hashtags"])
    post_type = row["post_type"]

    try:
        if post_type == "image":
            result = cl.photo_upload(str(media_path), caption=caption)
        elif post_type == "reel":
            thumbnail = None
            if row["thumbnail_path"]:
                thumb_path = UPLOADS_DIR / row["thumbnail_path"]
                if thumb_path.exists():
                    thumbnail = str(thumb_path)
            result = cl.clip_upload(str(media_path), caption=caption, thumbnail=thumbnail)
        elif post_type == "story":
            ext = media_path.suffix.lower()
            if ext in (".mp4", ".mov", ".avi"):
                result = cl.video_upload_to_story(str(media_path))
            else:
                result = cl.photo_upload_to_story(str(media_path))
        else:
            return False, f"Unknown post type: {post_type}"

        media_id = getattr(result, "pk", None) or getattr(result, "id", None)
        return True, str(media_id) if media_id else "unknown"

    except LoginRequired:
        if ig_relogin():
            return do_post(row, _resolved_challenge)
        return False, "Login expired and re-login failed"
    except (ChallengeRequired, ClientError) as e:
        last = getattr(cl, "last_json", {}) or {}
        if not _resolved_challenge and ("challenge" in last or "challenge_required" in str(e)):
            log.info("Challenge required during upload, resolving...")
            try:
                cl.challenge_resolve(last)
                cl.dump_settings(str(SESSION_FILE))
                log.info("Challenge resolved, retrying upload")
                return do_post(row, _resolved_challenge=True)
            except Exception as e2:
                return False, f"Challenge resolution failed: {e2}"
        return False, f"Instagram API error: {e}"
    except Exception as e:
        return False, f"Upload error: {e}"


# ── Queue processor ────────────────────────────────────────────────────────────

def process_queue():
    db = get_db()
    now = time.time()

    rows = db.execute(
        """SELECT * FROM posts
           WHERE status = 'pending' AND scheduled_at <= ?
           ORDER BY sort_order ASC, scheduled_at ASC
           LIMIT 1""",
        (now,),
    ).fetchall()

    if not rows:
        db.close()
        return

    row = rows[0]
    post_id = row["id"]
    log.info("Processing post #%d (%s): %s", post_id, row["post_type"], row["media_path"])

    db.execute("UPDATE posts SET status = 'posting' WHERE id = ?", (post_id,))
    db.commit()
    log_action(db, post_id, "posting", "Started upload")

    success, result = do_post(row)

    if success:
        db.execute(
            "UPDATE posts SET status = 'posted', instagram_media_id = ?, posted_at = ? WHERE id = ?",
            (result, time.time(), post_id),
        )
        db.commit()
        log_action(db, post_id, "posted", f"Media ID: {result}")
        log.info("Post #%d published successfully (media_id: %s)", post_id, result)
        cl.dump_settings(str(SESSION_FILE))

        # Queue story share if enabled
        if row["post_type"] in ("image", "reel") and row["share_to_story"]:
            auto_share = get_setting(db, "auto_share_story", "1")
            if auto_share == "1":
                db.execute("UPDATE posts SET story_status = 'pending' WHERE id = ?", (post_id,))
                db.commit()
                log.info("Post #%d queued for story sharing", post_id)
    else:
        retry_count = row["retry_count"] + 1
        if retry_count >= MAX_RETRIES:
            db.execute(
                "UPDATE posts SET status = 'failed', error_message = ?, retry_count = ? WHERE id = ?",
                (result, retry_count, post_id),
            )
            log_action(db, post_id, "failed", f"Max retries reached: {result}")
            log.error("Post #%d failed permanently after %d retries: %s", post_id, retry_count, result)
        else:
            db.execute(
                "UPDATE posts SET status = 'pending', error_message = ?, retry_count = ? WHERE id = ?",
                (result, retry_count, post_id),
            )
            log_action(db, post_id, "retry", f"Attempt {retry_count}: {result}")
            log.warning("Post #%d failed (attempt %d/%d): %s", post_id, retry_count, MAX_RETRIES, result)
        db.commit()

    db.close()


def process_story_shares():
    """Share any posted items that have story_status = 'pending'."""
    db = get_db()
    rows = db.execute(
        """SELECT * FROM posts
           WHERE status = 'posted' AND story_status = 'pending'
              AND instagram_media_id IS NOT NULL
           ORDER BY posted_at ASC LIMIT 1"""
    ).fetchall()

    if not rows:
        db.close()
        return

    row = rows[0]
    post_id = row["id"]
    log.info("Sharing post #%d to story...", post_id)

    shared = share_post_to_story(row["instagram_media_id"], row["media_path"])
    if shared:
        db.execute("UPDATE posts SET story_status = 'shared' WHERE id = ?", (post_id,))
        log_action(db, post_id, "story_shared", "Shared to story")
        log.info("Post #%d shared to story", post_id)
    else:
        db.execute("UPDATE posts SET story_status = 'failed' WHERE id = ?", (post_id,))
        log_action(db, post_id, "story_failed", "Story share failed")
        log.error("Post #%d story share failed", post_id)

    db.commit()
    db.close()


# ── Feed-to-story resharing ───────────────────────────────────────────────────

_cached_user_id = None


def get_own_user_id():
    global _cached_user_id
    if _cached_user_id:
        return _cached_user_id
    try:
        _cached_user_id = cl.user_id_from_username(IG_USERNAME)
        return _cached_user_id
    except Exception as e:
        log.error("Failed to get user ID: %s", e)
        return None


def process_feed_reshares():
    """Pick a random post from the user's feed and share it to story."""
    db = get_db()
    enabled = get_setting(db, "story_reshare_enabled", "0")
    if enabled != "1":
        db.close()
        return

    daily_count = max(1, int(get_setting(db, "story_reshare_count", "5")))
    interval = 86400.0 / daily_count  # seconds between shares

    # Check last reshare time
    last = db.execute(
        "SELECT shared_at FROM story_reshares ORDER BY shared_at DESC LIMIT 1"
    ).fetchone()
    now = time.time()
    if last and (now - last["shared_at"]) < interval:
        db.close()
        return

    # Count reshares today (since midnight)
    midnight = now - (now % 86400)
    today_count = db.execute(
        "SELECT COUNT(*) FROM story_reshares WHERE shared_at >= ?", (midnight,)
    ).fetchone()[0]
    if today_count >= daily_count:
        db.close()
        return

    # Get recent media IDs already reshared in the last 7 days to avoid repeats
    recent_ids = set()
    for r in db.execute(
        "SELECT instagram_media_id FROM story_reshares WHERE shared_at >= ?",
        (now - 7 * 86400,),
    ).fetchall():
        recent_ids.add(r["instagram_media_id"])

    db.close()

    # Fetch user's feed
    user_id = get_own_user_id()
    if not user_id:
        return

    log.info("Fetching feed for reshare (daily %d/%d)...", today_count + 1, daily_count)
    try:
        medias = cl.user_medias(user_id, amount=50)
    except Exception as e:
        log.error("Failed to fetch feed: %s", e)
        return

    if not medias:
        log.warning("No feed posts found for reshare")
        return

    # Filter out recently shared and stories
    candidates = [
        m for m in medias
        if str(m.pk) not in recent_ids and m.media_type in (1, 2, 8)
    ]
    if not candidates:
        log.info("No eligible posts for reshare (all recently shared)")
        return

    chosen = random.choice(candidates)
    media_pk = str(chosen.pk)
    log.info("Resharing feed post %s to story...", media_pk)

    try:
        story_media = StoryMedia(media_pk=int(media_pk))
        # Use the post's thumbnail URL to download a temp image for the story background
        thumb_url = None
        if chosen.thumbnail_url:
            thumb_url = str(chosen.thumbnail_url)
        elif chosen.image_versions2 and chosen.image_versions2.get("candidates"):
            thumb_url = chosen.image_versions2["candidates"][0]["url"]

        if thumb_url:
            # Download thumbnail to a temp file
            import tempfile
            import requests
            resp = requests.get(thumb_url, timeout=15)
            resp.raise_for_status()
            ext = ".jpg"
            tmp = tempfile.NamedTemporaryFile(suffix=ext, dir=str(UPLOADS_DIR), delete=False)
            tmp.write(resp.content)
            tmp.close()
            try:
                cl.photo_upload_to_story(tmp.name, medias=[story_media])
            finally:
                os.unlink(tmp.name)
        else:
            log.warning("No thumbnail available for post %s, skipping", media_pk)
            return

        log.info("Feed post %s reshared to story", media_pk)

        db = get_db()
        db.execute(
            "INSERT INTO story_reshares (instagram_media_id, shared_at) VALUES (?, ?)",
            (media_pk, time.time()),
        )
        db.commit()
        db.close()

    except Exception as e:
        log.error("Feed reshare failed: %s", e)


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    log.info("Instagram Autoposter starting")
    log.info("Username: %s", IG_USERNAME)
    log.info("Check interval: %ds", CHECK_INTERVAL)
    log.info("Max retries: %d", MAX_RETRIES)

    if not IG_USERNAME or not IG_PASSWORD:
        log.error("INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD must be set in .env")
        sys.exit(1)

    init_db()

    # Recover any posts stuck in 'posting' from a previous crash
    db = get_db()
    stuck = db.execute("UPDATE posts SET status = 'pending' WHERE status = 'posting'").rowcount
    if stuck:
        db.commit()
        log.info("Recovered %d post(s) stuck in 'posting' status", stuck)
    db.close()

    if not ig_login():
        log.error("Initial login failed — exiting")
        sys.exit(1)

    log.info("Poster daemon running. Checking queue every %ds...", CHECK_INTERVAL)

    while True:
        try:
            process_queue()
        except Exception as e:
            log.error("Queue processing error: %s", e, exc_info=True)

        try:
            process_story_shares()
        except Exception as e:
            log.error("Story share processing error: %s", e, exc_info=True)

        try:
            process_feed_reshares()
        except Exception as e:
            log.error("Feed reshare error: %s", e, exc_info=True)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
