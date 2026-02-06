"""
Instagram Autoposter Dashboard — Single-file Flask web dashboard.
Usage: python3 dashboard.py
"""

import os
import sqlite3
import subprocess
import time
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(32)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "instagram_autoposter.db"
UPLOADS_DIR = BASE_DIR / "uploads"
POSTER_LOG = BASE_DIR / "poster.log"
ENV_FILE = BASE_DIR / ".env"

UPLOADS_DIR.mkdir(exist_ok=True)

DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "admin")
PORT = int(os.getenv("DASHBOARD_PORT", "5556"))
POSTER_SERVICE = os.getenv("POSTER_SERVICE_LABEL", "com.instagram.autoposter")

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi"}
ALLOWED_EXT = ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT

EDITABLE_KEYS = [
    "INSTAGRAM_USERNAME", "INSTAGRAM_PASSWORD",
    "CHECK_INTERVAL", "MAX_RETRIES",
]


# ── DB helpers ─────────────────────────────────────────────────────────────────

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


init_db()


# ── Auth ───────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authed"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_launchd_status(label):
    try:
        result = subprocess.run(
            ["launchctl", "list", label], capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if '"PID"' in line or "PID" in line:
                    parts = line.strip().rstrip(";").split("=")
                    if len(parts) == 2:
                        return {"running": True, "pid": parts[1].strip().rstrip(";")}
            return {"running": True, "pid": "unknown"}
        return {"running": False, "pid": None}
    except Exception:
        return {"running": False, "pid": None}


def is_process_running(name):
    try:
        result = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True)
        pids = [p for p in result.stdout.strip().split("\n") if p]
        return pids if pids else None
    except Exception:
        return None


def tail_file(path, lines=50):
    try:
        result = subprocess.run(
            ["tail", "-n", str(lines), str(path)],
            capture_output=True, text=True
        )
        return result.stdout
    except Exception:
        return f"Could not read {path}"


def read_env():
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip()
    return env


def write_env(updates: dict):
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    updated_keys = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)
    for key, val in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={val}")
    ENV_FILE.write_text("\n".join(new_lines) + "\n")


def get_setting(key, default=""):
    db = get_db()
    row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    db.close()
    return row["value"] if row else default


def set_setting(key, value):
    db = get_db()
    db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    db.commit()
    db.close()


def ts_format(ts):
    if not ts:
        return "N/A"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def ts_format_short(ts):
    if not ts:
        return "N/A"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%m/%d %H:%M")
    except Exception:
        return str(ts)


def render_page(content_template, active="", **kwargs):
    flash_msg = session.pop("flash_msg", None)
    flash_type = session.pop("flash_type", "success")
    full = BASE_HTML.replace("{% block content %}{% endblock %}", content_template)
    full = full.replace("{% block title_extra %}{% endblock %}", kwargs.pop("title_extra", ""))
    full = full.replace("{% block scripts %}{% endblock %}", kwargs.pop("extra_scripts", ""))
    return render_template_string(
        full, active=active, flash_msg=flash_msg, flash_type=flash_type, **kwargs
    )


def flash(msg, type_="success"):
    session["flash_msg"] = msg
    session["flash_type"] = type_


def save_upload(file_storage):
    if not file_storage or not file_storage.filename:
        return None
    ext = Path(file_storage.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return None
    filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    file_storage.save(str(UPLOADS_DIR / filename))
    return filename


def delete_media(filename):
    if filename:
        path = UPLOADS_DIR / filename
        if path.exists():
            path.unlink()


# ── Base template ──────────────────────────────────────────────────────────────

BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IG Autoposter{% block title_extra %}{% endblock %}</title>
<style>
:root {
    --bg: #0d0d0d;
    --surface: #1a1a1a;
    --surface2: #242424;
    --border: #333;
    --text: #e0e0e0;
    --text2: #888;
    --accent: #7c5cbf;
    --accent2: #5b9bd5;
    --danger: #c0392b;
    --success: #27ae60;
    --warn: #f39c12;
    --info: #3498db;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
}
a { color: var(--accent2); text-decoration: none; }
a:hover { text-decoration: underline; }
.shell { display: flex; min-height: 100vh; }
.sidebar {
    width: 220px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 20px 0;
    position: fixed;
    top: 0; bottom: 0;
    overflow-y: auto;
}
.sidebar h1 { font-size: 18px; padding: 0 20px 16px; border-bottom: 1px solid var(--border); margin-bottom: 8px; color: var(--accent); letter-spacing: 1px; }
.sidebar a { display: block; padding: 10px 20px; color: var(--text2); font-size: 14px; border-left: 3px solid transparent; transition: all 0.15s; }
.sidebar a:hover, .sidebar a.active { background: var(--surface2); color: var(--text); border-left-color: var(--accent); text-decoration: none; }
.main { margin-left: 220px; flex: 1; padding: 24px 32px; min-width: 0; }
.page-title { font-size: 22px; font-weight: 600; margin-bottom: 20px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 16px; }
.card h3 { font-size: 14px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
.stat { text-align: center; padding: 16px; }
.stat .val { font-size: 28px; font-weight: 700; color: var(--accent); }
.stat .label { font-size: 12px; color: var(--text2); margin-top: 4px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
.badge-ok { background: rgba(39,174,96,0.2); color: var(--success); }
.badge-err { background: rgba(192,57,43,0.2); color: var(--danger); }
.badge-warn { background: rgba(243,156,18,0.2); color: var(--warn); }
.badge-info { background: rgba(52,152,219,0.2); color: var(--info); }
.badge-pending { background: rgba(136,136,136,0.2); color: var(--text2); }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th { text-align: left; padding: 10px 12px; border-bottom: 2px solid var(--border); color: var(--text2); font-size: 12px; text-transform: uppercase; }
td { padding: 10px 12px; border-bottom: 1px solid var(--border); }
tr:hover { background: var(--surface2); }
input[type="text"], input[type="password"], input[type="search"], input[type="number"],
input[type="datetime-local"], textarea, select {
    background: var(--surface2); border: 1px solid var(--border); color: var(--text);
    padding: 8px 12px; border-radius: 6px; font-size: 14px; width: 100%;
}
input[type="file"] { color: var(--text); font-size: 14px; }
input:focus, textarea:focus, select:focus { outline: none; border-color: var(--accent); }
.btn { display: inline-block; padding: 8px 16px; border-radius: 6px; border: none; font-size: 13px; font-weight: 600; cursor: pointer; transition: opacity 0.15s; }
.btn:hover { opacity: 0.85; }
.btn-primary { background: var(--accent); color: #fff; }
.btn-danger { background: var(--danger); color: #fff; }
.btn-warn { background: var(--warn); color: #000; }
.btn-success { background: var(--success); color: #fff; }
.btn-sm { padding: 4px 10px; font-size: 12px; }
.form-row { margin-bottom: 12px; }
.form-row label { display: block; font-size: 13px; color: var(--text2); margin-bottom: 4px; }
.log-viewer { background: #111; border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-family: 'SF Mono','Menlo',monospace; font-size: 12px; line-height: 1.6; white-space: pre-wrap; word-break: break-all; max-height: 500px; overflow-y: auto; color: #aaa; }
.alert { padding: 10px 16px; border-radius: 6px; margin-bottom: 16px; font-size: 14px; }
.alert-success { background: rgba(39,174,96,0.15); color: var(--success); border: 1px solid rgba(39,174,96,0.3); }
.alert-error { background: rgba(192,57,43,0.15); color: var(--danger); border: 1px solid rgba(192,57,43,0.3); }
.flex { display: flex; gap: 8px; align-items: center; }
.flex-between { display: flex; justify-content: space-between; align-items: center; }
.flex-wrap { flex-wrap: wrap; }
.mb-8 { margin-bottom: 8px; }
.mb-16 { margin-bottom: 16px; }
.mt-8 { margin-top: 8px; }
.mt-16 { margin-top: 16px; }
.section + .section { margin-top: 24px; }
.media-thumb { width: 60px; height: 60px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); }
.drag-handle { cursor: grab; color: var(--text2); user-select: none; }
.drag-handle:active { cursor: grabbing; }
.dragging { opacity: 0.4; }
@media (max-width: 768px) { .sidebar { display: none; } .main { margin-left: 0; padding: 16px; } .stats-grid { grid-template-columns: repeat(2, 1fr); } }
</style>
</head>
<body>
<div class="shell">
    <nav class="sidebar">
        <h1>IG POSTER</h1>
        <a href="/" class="{{ 'active' if active == 'overview' }}">Overview</a>
        <a href="/queue" class="{{ 'active' if active == 'queue' }}">Queue</a>
        <a href="/new" class="{{ 'active' if active == 'new' }}">New Post</a>
        <a href="/history" class="{{ 'active' if active == 'history' }}">History</a>
        <a href="/topics" class="{{ 'active' if active == 'topics' }}">Topics</a>
        <a href="/character" class="{{ 'active' if active == 'character' }}">Character</a>
        <a href="/generated" class="{{ 'active' if active == 'generated' }}">Generated</a>
        <a href="/settings" class="{{ 'active' if active == 'settings' }}">Settings</a>
    </nav>
    <div class="main">
        {% if flash_msg %}
        <div class="alert alert-{{ flash_type|default('success') }}">{{ flash_msg }}</div>
        {% endif %}
        {% block content %}{% endblock %}
    </div>
</div>
{% block scripts %}{% endblock %}
</body>
</html>"""


# ── Login ──────────────────────────────────────────────────────────────────────

LOGIN_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IG Autoposter - Login</title>
<style>
body { background: #0d0d0d; color: #e0e0e0; font-family: -apple-system, system-ui, sans-serif;
       display: flex; align-items: center; justify-content: center; min-height: 100vh; }
.login-box { background: #1a1a1a; border: 1px solid #333; border-radius: 12px; padding: 40px;
             width: 320px; text-align: center; }
.login-box h1 { color: #7c5cbf; margin-bottom: 24px; font-size: 20px; letter-spacing: 1px; }
.login-box input { background: #242424; border: 1px solid #333; color: #e0e0e0; padding: 10px 14px;
                   border-radius: 6px; width: 100%; font-size: 14px; margin-bottom: 16px; }
.login-box input:focus { outline: none; border-color: #7c5cbf; }
.login-box button { background: #7c5cbf; color: #fff; border: none; padding: 10px 24px;
                    border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; width: 100%; }
.login-box button:hover { opacity: 0.85; }
.err { color: #c0392b; font-size: 13px; margin-bottom: 12px; }
</style></head><body>
<div class="login-box">
    <h1>IG AUTOPOSTER</h1>
    {% if error %}<div class="err">{{ error }}</div>{% endif %}
    <form method="POST">
        <input type="password" name="password" placeholder="Password" autofocus>
        <button type="submit">Enter</button>
    </form>
</div>
</body></html>"""


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == DASHBOARD_PASSWORD:
            session["authed"] = True
            return redirect("/")
        error = "Wrong password"
    return render_template_string(LOGIN_HTML, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ── Uploads serving ────────────────────────────────────────────────────────────

@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(str(UPLOADS_DIR), filename)


# ── Overview ───────────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def overview():
    db = get_db()
    total = db.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    posted = db.execute("SELECT COUNT(*) FROM posts WHERE status = 'posted'").fetchone()[0]
    pending = db.execute("SELECT COUNT(*) FROM posts WHERE status = 'pending'").fetchone()[0]
    failed = db.execute("SELECT COUNT(*) FROM posts WHERE status = 'failed'").fetchone()[0]
    generated_ready = db.execute("SELECT COUNT(*) FROM generation_sessions WHERE status = 'content_ready'").fetchone()[0]

    upcoming = db.execute(
        "SELECT * FROM posts WHERE status = 'pending' ORDER BY scheduled_at ASC LIMIT 5"
    ).fetchall()

    recent = db.execute(
        "SELECT * FROM posts WHERE status = 'posted' ORDER BY posted_at DESC LIMIT 5"
    ).fetchall()

    recent_sessions = db.execute(
        "SELECT gs.*, t.name as topic_name FROM generation_sessions gs LEFT JOIN topics t ON gs.topic_id = t.id ORDER BY gs.created_at DESC LIMIT 10"
    ).fetchall()
    db.close()

    poster_status = get_launchd_status(POSTER_SERVICE)
    poster_pids = is_process_running("poster.py")
    generator_pids = is_process_running("generator.py")
    generator_log = tail_file(BASE_DIR / "generator.log", 30)
    poster_log = tail_file(POSTER_LOG, 30)

    content = """
<div class="page-title">Overview</div>
<div class="stats-grid">
    <div class="card stat"><div class="val">{{ total }}</div><div class="label">Total Posts</div></div>
    <div class="card stat"><div class="val">{{ posted }}</div><div class="label">Posted</div></div>
    <div class="card stat"><div class="val">{{ pending }}</div><div class="label">Pending</div></div>
    <div class="card stat"><div class="val">{{ failed }}</div><div class="label">Failed</div></div>
    <div class="card stat"><div class="val">{{ generated_ready }}</div><div class="label">Generated</div></div>
</div>

<div class="card section">
    <h3>Service Status</h3>
    <table>
        <tr>
            <td style="width:200px">Poster Daemon</td>
            <td>
                {% if poster_status.running %}
                    <span class="badge badge-ok">Running</span> PID {{ poster_status.pid }}
                {% elif poster_pids %}
                    <span class="badge badge-ok">Running</span> PID {{ poster_pids[0] }} (manual)
                {% else %}
                    <span class="badge badge-err">Stopped</span>
                {% endif %}
            </td>
            <td style="width:100px">
                <form method="POST" action="/service/restart" style="display:inline">
                    <button class="btn btn-sm btn-warn" type="submit">Restart</button>
                </form>
            </td>
        </tr>
        <tr>
            <td>Generator Bot</td>
            <td>
                {% if generator_pids %}
                    <span class="badge badge-ok">Running</span> PID {{ generator_pids[0] }}
                {% else %}
                    <span class="badge badge-err">Stopped</span>
                {% endif %}
            </td>
            <td></td>
        </tr>
    </table>
</div>

{% if upcoming %}
<div class="card section">
    <h3>Upcoming Posts</h3>
    <table>
        <thead><tr><th>Type</th><th>Caption</th><th>Scheduled</th></tr></thead>
        <tbody>
        {% for p in upcoming %}
        <tr>
            <td><span class="badge badge-info">{{ p.post_type }}</span></td>
            <td>{{ p.caption[:80] if p.caption else '(no caption)' }}{% if p.caption and p.caption|length > 80 %}...{% endif %}</td>
            <td style="white-space:nowrap">{{ ts_format(p.scheduled_at) }}</td>
        </tr>
        {% endfor %}
        </tbody>
    </table>
</div>
{% endif %}

{% if recent %}
<div class="card section">
    <h3>Recent Posts</h3>
    <table>
        <thead><tr><th>Type</th><th>Caption</th><th>Posted</th></tr></thead>
        <tbody>
        {% for p in recent %}
        <tr>
            <td><span class="badge badge-ok">{{ p.post_type }}</span></td>
            <td>{{ p.caption[:80] if p.caption else '(no caption)' }}{% if p.caption and p.caption|length > 80 %}...{% endif %}</td>
            <td style="white-space:nowrap">{{ ts_format(p.posted_at) }}</td>
        </tr>
        {% endfor %}
        </tbody>
    </table>
</div>
{% endif %}

{% if recent_sessions %}
<div class="card section">
    <h3>Recent Generations</h3>
    <table>
        <thead><tr><th>Topic</th><th>Type</th><th>Status</th><th>Time</th><th>Error</th></tr></thead>
        <tbody>
        {% for s in recent_sessions %}
        <tr>
            <td>{{ s.topic_name or 'Unknown' }}</td>
            <td><span class="badge badge-info">{{ s.generated_media_type or '—' }}</span></td>
            <td>
                {% if s.status == 'content_ready' %}<span class="badge badge-ok">Ready</span>
                {% elif s.status == 'inserted' %}<span class="badge badge-ok">Inserted</span>
                {% elif s.status == 'generating' %}<span class="badge badge-warn">Generating</span>
                {% elif s.status == 'awaiting_response' %}<span class="badge badge-warn">Awaiting</span>
                {% elif s.status == 'failed' %}<span class="badge badge-err">Failed</span>
                {% elif s.status == 'cancelled' %}<span class="badge" style="background:#555">Cancelled</span>
                {% else %}<span class="badge">{{ s.status }}</span>
                {% endif %}
            </td>
            <td style="white-space:nowrap">{{ ts_format(s.created_at) }}</td>
            <td style="color:#e55;font-size:12px">{{ s.error_message[:60] if s.error_message else '' }}</td>
        </tr>
        {% endfor %}
        </tbody>
    </table>
</div>
{% endif %}

<div class="card section">
    <h3>Generator Log</h3>
    <div class="log-viewer">{{ generator_log }}</div>
</div>

<div class="card section">
    <h3>Poster Log</h3>
    <div class="log-viewer">{{ poster_log }}</div>
</div>
"""
    return render_page(
        content, active="overview",
        total=total, posted=posted, pending=pending, failed=failed, generated_ready=generated_ready,
        upcoming=upcoming, recent=recent, recent_sessions=recent_sessions,
        poster_status=poster_status, poster_pids=poster_pids,
        generator_pids=generator_pids,
        generator_log=generator_log, poster_log=poster_log, ts_format=ts_format,
    )


# ── Queue ──────────────────────────────────────────────────────────────────────

@app.route("/queue")
@login_required
def queue():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM posts WHERE status IN ('pending', 'posting') ORDER BY sort_order ASC, scheduled_at ASC"
    ).fetchall()
    db.close()

    content = """
<div class="page-title">Queue</div>

<div class="card">
    <div class="flex-between mb-8">
        <h3 style="margin:0">{{ rows|length }} post{{ 's' if rows|length != 1 }} in queue</h3>
        <a href="/new" class="btn btn-primary btn-sm">+ New Post</a>
    </div>

    {% if rows %}
    <table id="queue-table">
        <thead><tr><th></th><th>Preview</th><th>Type</th><th>Caption</th><th>Scheduled</th><th>Status</th><th>Actions</th></tr></thead>
        <tbody>
        {% for p in rows %}
        <tr data-id="{{ p.id }}" draggable="true">
            <td class="drag-handle">&#x2630;</td>
            <td>
                {% if p.post_type == 'reel' %}
                    <div style="width:60px;height:60px;background:var(--surface2);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:20px;border:1px solid var(--border)">&#9654;</div>
                {% else %}
                    <img src="/uploads/{{ p.media_path }}" class="media-thumb" onerror="this.style.display='none'">
                {% endif %}
            </td>
            <td><span class="badge badge-info">{{ p.post_type }}</span></td>
            <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis">{{ p.caption[:60] if p.caption else '(no caption)' }}{% if p.caption and p.caption|length > 60 %}...{% endif %}</td>
            <td style="white-space:nowrap">{{ ts_format(p.scheduled_at) }}</td>
            <td>
                {% if p.status == 'posting' %}
                    <span class="badge badge-warn">Posting</span>
                {% else %}
                    <span class="badge badge-pending">Pending</span>
                {% endif %}
                {% if p.retry_count > 0 %}
                    <span style="font-size:11px;color:var(--warn)">(retry {{ p.retry_count }})</span>
                {% endif %}
            </td>
            <td style="white-space:nowrap">
                <div class="flex">
                    <form method="POST" action="/queue/post-now/{{ p.id }}">
                        <button class="btn btn-sm btn-success" type="submit">Post Now</button>
                    </form>
                    <a href="/edit/{{ p.id }}" class="btn btn-sm btn-primary">Edit</a>
                    <form method="POST" action="/queue/delete/{{ p.id }}" onsubmit="return confirm('Delete this post?')">
                        <button class="btn btn-sm btn-danger" type="submit">Delete</button>
                    </form>
                </div>
            </td>
        </tr>
        {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p style="color:var(--text2)">Queue is empty. <a href="/new">Create a post</a> to get started.</p>
    {% endif %}
</div>
"""

    extra_scripts = """
<script>
(function() {
    const tbody = document.querySelector('#queue-table tbody');
    if (!tbody) return;
    let dragRow = null;

    tbody.addEventListener('dragstart', function(e) {
        dragRow = e.target.closest('tr');
        if (dragRow) dragRow.classList.add('dragging');
    });

    tbody.addEventListener('dragover', function(e) {
        e.preventDefault();
        const target = e.target.closest('tr');
        if (target && target !== dragRow) {
            const rect = target.getBoundingClientRect();
            const mid = rect.top + rect.height / 2;
            if (e.clientY < mid) {
                tbody.insertBefore(dragRow, target);
            } else {
                tbody.insertBefore(dragRow, target.nextSibling);
            }
        }
    });

    tbody.addEventListener('dragend', function() {
        if (dragRow) dragRow.classList.remove('dragging');
        dragRow = null;
        // Save new order
        const ids = Array.from(tbody.querySelectorAll('tr')).map(r => r.dataset.id);
        fetch('/queue/reorder', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({order: ids})
        });
    });
})();
</script>
"""
    return render_page(content, active="queue", rows=rows, ts_format=ts_format, extra_scripts=extra_scripts)


@app.route("/queue/reorder", methods=["POST"])
@login_required
def queue_reorder():
    data = request.get_json()
    if not data or "order" not in data:
        return jsonify({"error": "Missing order"}), 400
    db = get_db()
    for idx, post_id in enumerate(data["order"]):
        db.execute("UPDATE posts SET sort_order = ? WHERE id = ?", (idx, int(post_id)))
    db.commit()
    db.close()
    return jsonify({"ok": True})


@app.route("/queue/post-now/<int:post_id>", methods=["POST"])
@login_required
def queue_post_now(post_id):
    db = get_db()
    db.execute(
        "UPDATE posts SET scheduled_at = ?, status = 'pending', retry_count = 0 WHERE id = ? AND status IN ('pending', 'posting', 'failed')",
        (time.time() - 1, post_id),
    )
    db.commit()
    db.close()
    flash("Post scheduled for immediate publishing")
    return redirect("/queue")


@app.route("/queue/delete/<int:post_id>", methods=["POST"])
@login_required
def queue_delete(post_id):
    db = get_db()
    row = db.execute("SELECT media_path, thumbnail_path FROM posts WHERE id = ?", (post_id,)).fetchone()
    if row:
        delete_media(row["media_path"])
        delete_media(row["thumbnail_path"])
        db.execute("DELETE FROM posts WHERE id = ?", (post_id,))
        db.execute("DELETE FROM post_log WHERE post_id = ?", (post_id,))
        db.commit()
        flash("Post deleted")
    db.close()
    return redirect("/queue")


# ── New Post ───────────────────────────────────────────────────────────────────

@app.route("/new", methods=["GET"])
@login_required
def new_post():
    default_hashtags = get_setting("default_hashtags")
    auto_share_story = get_setting("auto_share_story", "1")

    content = """
<div class="page-title">New Post</div>

<div class="card">
    <form method="POST" action="/new" enctype="multipart/form-data">
        <div class="form-row">
            <label>Post Type</label>
            <select name="post_type" id="post-type" style="width:200px" onchange="toggleFields()">
                <option value="image">Image</option>
                <option value="reel">Reel (Video)</option>
                <option value="story">Story</option>
            </select>
        </div>

        <div class="form-row">
            <label>Media File</label>
            <input type="file" name="media" accept="image/*,video/*" required>
        </div>

        <div class="form-row" id="thumbnail-row" style="display:none">
            <label>Thumbnail (optional, for reels)</label>
            <input type="file" name="thumbnail" accept="image/*">
        </div>

        <div class="form-row">
            <label>Caption</label>
            <textarea name="caption" rows="4" placeholder="Write your caption..."></textarea>
        </div>

        <div class="form-row">
            <label>Hashtags</label>
            <textarea name="hashtags" rows="2" placeholder="#hashtag1 #hashtag2">{{ default_hashtags }}</textarea>
        </div>

        <div class="form-row">
            <label>Schedule For</label>
            <input type="datetime-local" name="scheduled_at" style="width:300px">
            <span style="font-size:12px;color:var(--text2);margin-top:4px;display:block">Leave empty to post as soon as possible</span>
        </div>

        <div class="form-row" id="share-story-row">
            <label style="display:inline">
                <input type="checkbox" name="share_to_story" value="1" {{ 'checked' if auto_share_story == '1' }} style="width:auto">
                Also share to story
            </label>
            <span style="font-size:12px;color:var(--text2);display:block;margin-top:2px">Automatically reshare this post to your story after publishing</span>
        </div>

        <button class="btn btn-primary mt-8" type="submit">Add to Queue</button>
    </form>
</div>

<script>
function toggleFields() {
    var type = document.getElementById('post-type').value;
    document.getElementById('thumbnail-row').style.display = type === 'reel' ? 'block' : 'none';
    document.getElementById('share-story-row').style.display = type === 'story' ? 'none' : 'block';
}
</script>
"""
    return render_page(content, active="new", default_hashtags=default_hashtags, auto_share_story=auto_share_story)


@app.route("/new", methods=["POST"])
@login_required
def new_post_submit():
    post_type = request.form.get("post_type", "image")
    caption = request.form.get("caption", "").strip()
    hashtags = request.form.get("hashtags", "").strip()
    scheduled_str = request.form.get("scheduled_at", "").strip()

    media_file = request.files.get("media")
    if not media_file or not media_file.filename:
        flash("Media file is required", "error")
        return redirect("/new")

    media_filename = save_upload(media_file)
    if not media_filename:
        flash("Invalid file type. Allowed: jpg, png, webp, mp4, mov, avi", "error")
        return redirect("/new")

    thumbnail_filename = None
    if post_type == "reel":
        thumb_file = request.files.get("thumbnail")
        if thumb_file and thumb_file.filename:
            thumbnail_filename = save_upload(thumb_file)

    if scheduled_str:
        try:
            dt = datetime.strptime(scheduled_str, "%Y-%m-%dT%H:%M")
            scheduled_at = dt.timestamp()
        except ValueError:
            scheduled_at = time.time()
    else:
        scheduled_at = time.time()

    share_to_story = 1 if request.form.get("share_to_story") else 0

    db = get_db()
    max_order = db.execute("SELECT COALESCE(MAX(sort_order), 0) FROM posts WHERE status = 'pending'").fetchone()[0]
    db.execute(
        """INSERT INTO posts (post_type, media_path, thumbnail_path, caption, hashtags,
           status, scheduled_at, sort_order, share_to_story, created_at)
           VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)""",
        (post_type, media_filename, thumbnail_filename, caption, hashtags,
         scheduled_at, max_order + 1, share_to_story, time.time()),
    )
    post_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    db.execute(
        "INSERT INTO post_log (post_id, action, details, timestamp) VALUES (?, 'queued', 'Added via dashboard', ?)",
        (post_id, time.time()),
    )
    db.commit()
    db.close()

    flash(f"Post #{post_id} added to queue")
    return redirect("/queue")


# ── Edit Post ──────────────────────────────────────────────────────────────────

@app.route("/edit/<int:post_id>", methods=["GET"])
@login_required
def edit_post(post_id):
    db = get_db()
    post = db.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
    db.close()
    if not post:
        flash("Post not found", "error")
        return redirect("/queue")

    scheduled_dt = ""
    if post["scheduled_at"]:
        try:
            scheduled_dt = datetime.fromtimestamp(post["scheduled_at"]).strftime("%Y-%m-%dT%H:%M")
        except Exception:
            pass

    content = """
<div class="page-title">Edit Post #{{ post.id }}</div>

<div class="card">
    <form method="POST" action="/edit/{{ post.id }}" enctype="multipart/form-data">
        <div class="form-row">
            <label>Post Type</label>
            <select name="post_type" style="width:200px">
                <option value="image" {{ 'selected' if post.post_type == 'image' }}>Image</option>
                <option value="reel" {{ 'selected' if post.post_type == 'reel' }}>Reel (Video)</option>
                <option value="story" {{ 'selected' if post.post_type == 'story' }}>Story</option>
            </select>
        </div>

        <div class="form-row">
            <label>Current Media: {{ post.media_path }}</label>
            {% if post.post_type != 'reel' %}
                <img src="/uploads/{{ post.media_path }}" style="max-width:200px;border-radius:6px;margin:8px 0" onerror="this.style.display='none'">
            {% endif %}
        </div>

        <div class="form-row">
            <label>Replace Media (optional)</label>
            <input type="file" name="media" accept="image/*,video/*">
        </div>

        <div class="form-row">
            <label>Caption</label>
            <textarea name="caption" rows="4">{{ post.caption or '' }}</textarea>
        </div>

        <div class="form-row">
            <label>Hashtags</label>
            <textarea name="hashtags" rows="2">{{ post.hashtags or '' }}</textarea>
        </div>

        <div class="form-row">
            <label>Schedule For</label>
            <input type="datetime-local" name="scheduled_at" value="{{ scheduled_dt }}" style="width:300px">
        </div>

        {% if post.post_type != 'story' %}
        <div class="form-row">
            <label style="display:inline">
                <input type="checkbox" name="share_to_story" value="1" {{ 'checked' if post.share_to_story }} style="width:auto">
                Also share to story
            </label>
        </div>
        {% endif %}

        <div class="flex mt-8">
            <button class="btn btn-primary" type="submit">Save Changes</button>
            <a href="/queue" class="btn btn-sm" style="background:var(--surface2);color:var(--text)">Cancel</a>
        </div>
    </form>
</div>
"""
    return render_page(content, active="queue", post=post, scheduled_dt=scheduled_dt)


@app.route("/edit/<int:post_id>", methods=["POST"])
@login_required
def edit_post_submit(post_id):
    db = get_db()
    post = db.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
    if not post:
        db.close()
        flash("Post not found", "error")
        return redirect("/queue")

    post_type = request.form.get("post_type", post["post_type"])
    caption = request.form.get("caption", "").strip()
    hashtags = request.form.get("hashtags", "").strip()
    scheduled_str = request.form.get("scheduled_at", "").strip()

    media_filename = post["media_path"]
    new_media = request.files.get("media")
    if new_media and new_media.filename:
        new_filename = save_upload(new_media)
        if new_filename:
            delete_media(post["media_path"])
            media_filename = new_filename

    if scheduled_str:
        try:
            dt = datetime.strptime(scheduled_str, "%Y-%m-%dT%H:%M")
            scheduled_at = dt.timestamp()
        except ValueError:
            scheduled_at = post["scheduled_at"]
    else:
        scheduled_at = post["scheduled_at"]

    share_to_story = 1 if request.form.get("share_to_story") else 0

    db.execute(
        """UPDATE posts SET post_type = ?, media_path = ?, caption = ?, hashtags = ?,
           scheduled_at = ?, share_to_story = ? WHERE id = ?""",
        (post_type, media_filename, caption, hashtags, scheduled_at, share_to_story, post_id),
    )
    db.commit()
    db.close()

    flash(f"Post #{post_id} updated")
    return redirect("/queue")


# ── History ────────────────────────────────────────────────────────────────────

@app.route("/history")
@login_required
def history():
    db = get_db()
    status_filter = request.args.get("status", "")
    type_filter = request.args.get("type", "")
    page_num = max(1, int(request.args.get("page", 1)))
    per_page = 30

    where = ["status IN ('posted', 'failed')"]
    params = []
    if status_filter:
        where = [f"status = ?"]
        params.append(status_filter)
    if type_filter:
        where.append("post_type = ?")
        params.append(type_filter)

    where_clause = " AND ".join(where)
    total = db.execute(f"SELECT COUNT(*) FROM posts WHERE {where_clause}", params).fetchone()[0]
    rows = db.execute(
        f"""SELECT * FROM posts WHERE {where_clause}
            ORDER BY COALESCE(posted_at, created_at) DESC LIMIT ? OFFSET ?""",
        params + [per_page, (page_num - 1) * per_page],
    ).fetchall()
    db.close()

    total_pages = max(1, (total + per_page - 1) // per_page)

    content = """
<div class="page-title">History</div>

<div class="card section">
    <form method="GET" action="/history">
        <div class="flex">
            <select name="status" style="width:150px">
                <option value="">All Statuses</option>
                <option value="posted" {{ 'selected' if status_filter == 'posted' }}>Posted</option>
                <option value="failed" {{ 'selected' if status_filter == 'failed' }}>Failed</option>
            </select>
            <select name="type" style="width:150px">
                <option value="">All Types</option>
                <option value="image" {{ 'selected' if type_filter == 'image' }}>Image</option>
                <option value="reel" {{ 'selected' if type_filter == 'reel' }}>Reel</option>
                <option value="story" {{ 'selected' if type_filter == 'story' }}>Story</option>
            </select>
            <button class="btn btn-primary" type="submit">Filter</button>
        </div>
    </form>
</div>

<div class="card section">
    <div class="flex-between mb-8">
        <h3 style="margin:0">{{ total }} post{{ 's' if total != 1 }}</h3>
        <div class="flex" style="font-size:13px;color:var(--text2)">
            Page {{ page_num }} of {{ total_pages }}
            {% if page_num > 1 %}<a href="?status={{ status_filter }}&type={{ type_filter }}&page={{ page_num-1 }}">&laquo; Prev</a>{% endif %}
            {% if page_num < total_pages %}<a href="?status={{ status_filter }}&type={{ type_filter }}&page={{ page_num+1 }}">Next &raquo;</a>{% endif %}
        </div>
    </div>

    {% if rows %}
    <table>
        <thead><tr><th>Preview</th><th>Type</th><th>Caption</th><th>Status</th><th>Date</th><th>Actions</th></tr></thead>
        <tbody>
        {% for p in rows %}
        <tr>
            <td>
                {% if p.post_type == 'reel' %}
                    <div style="width:50px;height:50px;background:var(--surface2);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:16px;border:1px solid var(--border)">&#9654;</div>
                {% else %}
                    <img src="/uploads/{{ p.media_path }}" class="media-thumb" style="width:50px;height:50px" onerror="this.style.display='none'">
                {% endif %}
            </td>
            <td><span class="badge badge-info">{{ p.post_type }}</span></td>
            <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis">{{ p.caption[:60] if p.caption else '(no caption)' }}{% if p.caption and p.caption|length > 60 %}...{% endif %}</td>
            <td>
                {% if p.status == 'posted' %}
                    <span class="badge badge-ok">Posted</span>
                {% else %}
                    <span class="badge badge-err">Failed</span>
                    {% if p.error_message %}
                    <div style="font-size:11px;color:var(--danger);margin-top:4px;max-width:200px">{{ p.error_message[:100] }}</div>
                    {% endif %}
                {% endif %}
            </td>
            <td style="white-space:nowrap">{{ ts_format(p.posted_at or p.created_at) }}</td>
            <td style="white-space:nowrap">
                <div class="flex">
                {% if p.status == 'failed' %}
                <form method="POST" action="/history/retry/{{ p.id }}">
                    <button class="btn btn-sm btn-warn" type="submit">Retry</button>
                </form>
                {% endif %}
                {% if p.status == 'posted' and p.post_type in ('image', 'reel') %}
                <form method="POST" action="/history/share-story/{{ p.id }}">
                    <button class="btn btn-sm btn-primary" type="submit">Share to Story</button>
                </form>
                {% endif %}
                </div>
            </td>
        </tr>
        {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p style="color:var(--text2)">No posts found.</p>
    {% endif %}
</div>
"""
    return render_page(
        content, active="history",
        rows=rows, total=total, page_num=page_num, total_pages=total_pages,
        status_filter=status_filter, type_filter=type_filter, ts_format=ts_format,
    )


@app.route("/history/retry/<int:post_id>", methods=["POST"])
@login_required
def history_retry(post_id):
    db = get_db()
    db.execute(
        "UPDATE posts SET status = 'pending', error_message = NULL, retry_count = 0, scheduled_at = ? WHERE id = ? AND status = 'failed'",
        (time.time(), post_id),
    )
    db.commit()
    db.execute(
        "INSERT INTO post_log (post_id, action, details, timestamp) VALUES (?, 'retry', 'Manual retry from dashboard', ?)",
        (post_id, time.time()),
    )
    db.commit()
    db.close()
    flash(f"Post #{post_id} queued for retry")
    return redirect("/history")


@app.route("/history/share-story/<int:post_id>", methods=["POST"])
@login_required
def history_share_story(post_id):
    db = get_db()
    post = db.execute(
        "SELECT id, status, instagram_media_id, post_type FROM posts WHERE id = ?", (post_id,)
    ).fetchone()
    if not post or post["status"] != "posted" or not post["instagram_media_id"]:
        flash("Cannot share this post to story", "error")
        db.close()
        return redirect("/history")
    db.execute("UPDATE posts SET story_status = 'pending' WHERE id = ?", (post_id,))
    db.execute(
        "INSERT INTO post_log (post_id, action, details, timestamp) VALUES (?, 'story_queued', 'Manual story share from dashboard', ?)",
        (post_id, time.time()),
    )
    db.commit()
    db.close()
    flash(f"Post #{post_id} queued for story sharing")
    return redirect("/history")


# ── Topics ────────────────────────────────────────────────────────────────────

@app.route("/topics")
@login_required
def topics():
    db = get_db()
    rows = db.execute("SELECT * FROM topics ORDER BY sort_order ASC, id ASC").fetchall()
    db.close()

    content = """
<div class="page-title">Topics</div>

<div class="card">
    <div class="flex-between mb-8">
        <h3 style="margin:0">{{ rows|length }} topic{{ 's' if rows|length != 1 }}</h3>
        <a href="/topics/new" class="btn btn-primary btn-sm">+ New Topic</a>
    </div>

    {% if rows %}
    <table>
        <thead><tr><th>Name</th><th>Description</th><th>Enabled</th><th>Last Asked</th><th>Actions</th></tr></thead>
        <tbody>
        {% for t in rows %}
        <tr>
            <td>{{ t.name }}</td>
            <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis">{{ t.description[:80] if t.description else '' }}{% if t.description and t.description|length > 80 %}...{% endif %}</td>
            <td>
                <form method="POST" action="/topics/toggle/{{ t.id }}" style="display:inline">
                    {% if t.enabled %}
                        <button class="btn btn-sm btn-success" type="submit">On</button>
                    {% else %}
                        <button class="btn btn-sm btn-danger" type="submit">Off</button>
                    {% endif %}
                </form>
            </td>
            <td style="white-space:nowrap">{{ ts_format(t.last_asked_at) }}</td>
            <td style="white-space:nowrap">
                <div class="flex">
                    <a href="/generate/new?topic_id={{ t.id }}" class="btn btn-sm btn-success">Generate</a>
                    <a href="/topics/edit/{{ t.id }}" class="btn btn-sm btn-primary">Edit</a>
                    <form method="POST" action="/topics/delete/{{ t.id }}" onsubmit="return confirm('Delete this topic?')">
                        <button class="btn btn-sm btn-danger" type="submit">Delete</button>
                    </form>
                </div>
            </td>
        </tr>
        {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p style="color:var(--text2)">No topics yet. <a href="/topics/new">Create one</a> to get started.</p>
    {% endif %}
</div>
"""
    return render_page(content, active="topics", rows=rows, ts_format=ts_format)


@app.route("/topics/new", methods=["GET"])
@login_required
def topics_new():
    content = """
<div class="page-title">New Topic</div>

<div class="card">
    <form method="POST" action="/topics/new">
        <div class="form-row">
            <label>Name</label>
            <input type="text" name="name" required style="width:400px" placeholder="e.g. Mindfulness">
        </div>
        <div class="form-row">
            <label>Description</label>
            <textarea name="description" rows="3" style="width:400px" placeholder="What this topic covers..."></textarea>
        </div>
        <div class="form-row">
            <label>Question Prompt (optional)</label>
            <textarea name="question_prompt" rows="3" style="width:400px" placeholder="Custom instruction for generating questions about this topic..."></textarea>
            <span style="font-size:12px;color:var(--text2);display:block;margin-top:2px">Leave empty to use default question generation</span>
        </div>
        <div class="flex mt-8">
            <button class="btn btn-primary" type="submit">Create Topic</button>
            <a href="/topics" class="btn btn-sm" style="background:var(--surface2);color:var(--text)">Cancel</a>
        </div>
    </form>
</div>
"""
    return render_page(content, active="topics")


@app.route("/topics/new", methods=["POST"])
@login_required
def topics_new_submit():
    name = request.form.get("name", "").strip()
    if not name:
        flash("Topic name is required", "error")
        return redirect("/topics/new")
    description = request.form.get("description", "").strip()
    question_prompt = request.form.get("question_prompt", "").strip()
    db = get_db()
    max_order = db.execute("SELECT COALESCE(MAX(sort_order), 0) FROM topics").fetchone()[0]
    db.execute(
        "INSERT INTO topics (name, description, question_prompt, sort_order, created_at) VALUES (?, ?, ?, ?, ?)",
        (name, description, question_prompt, max_order + 1, time.time()),
    )
    db.commit()
    db.close()
    flash(f"Topic '{name}' created")
    return redirect("/topics")


@app.route("/topics/edit/<int:topic_id>", methods=["GET"])
@login_required
def topics_edit(topic_id):
    db = get_db()
    topic = db.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
    db.close()
    if not topic:
        flash("Topic not found", "error")
        return redirect("/topics")

    content = """
<div class="page-title">Edit Topic: {{ topic.name }}</div>

<div class="card">
    <form method="POST" action="/topics/edit/{{ topic.id }}">
        <div class="form-row">
            <label>Name</label>
            <input type="text" name="name" value="{{ topic.name }}" required style="width:400px">
        </div>
        <div class="form-row">
            <label>Description</label>
            <textarea name="description" rows="3" style="width:400px">{{ topic.description or '' }}</textarea>
        </div>
        <div class="form-row">
            <label>Question Prompt (optional)</label>
            <textarea name="question_prompt" rows="3" style="width:400px">{{ topic.question_prompt or '' }}</textarea>
            <span style="font-size:12px;color:var(--text2);display:block;margin-top:2px">Leave empty to use default question generation</span>
        </div>
        <div class="flex mt-8">
            <button class="btn btn-primary" type="submit">Save Changes</button>
            <a href="/topics" class="btn btn-sm" style="background:var(--surface2);color:var(--text)">Cancel</a>
        </div>
    </form>
</div>
"""
    return render_page(content, active="topics", topic=topic)


@app.route("/topics/edit/<int:topic_id>", methods=["POST"])
@login_required
def topics_edit_submit(topic_id):
    name = request.form.get("name", "").strip()
    if not name:
        flash("Topic name is required", "error")
        return redirect(f"/topics/edit/{topic_id}")
    description = request.form.get("description", "").strip()
    question_prompt = request.form.get("question_prompt", "").strip()
    db = get_db()
    db.execute(
        "UPDATE topics SET name = ?, description = ?, question_prompt = ? WHERE id = ?",
        (name, description, question_prompt, topic_id),
    )
    db.commit()
    db.close()
    flash(f"Topic '{name}' updated")
    return redirect("/topics")


@app.route("/topics/delete/<int:topic_id>", methods=["POST"])
@login_required
def topics_delete(topic_id):
    db = get_db()
    db.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
    db.commit()
    db.close()
    flash("Topic deleted")
    return redirect("/topics")


@app.route("/topics/toggle/<int:topic_id>", methods=["POST"])
@login_required
def topics_toggle(topic_id):
    db = get_db()
    topic = db.execute("SELECT enabled FROM topics WHERE id = ?", (topic_id,)).fetchone()
    if topic:
        new_val = 0 if topic["enabled"] else 1
        db.execute("UPDATE topics SET enabled = ? WHERE id = ?", (new_val, topic_id))
        db.commit()
    db.close()
    return redirect("/topics")


# ── Character Config ──────────────────────────────────────────────────────────

@app.route("/character")
@login_required
def character():
    db = get_db()
    rows = db.execute("SELECT key, value FROM character_config").fetchall()
    db.close()
    config = {r["key"]: r["value"] for r in rows}

    content = """
<div class="page-title">Character Config</div>

<div class="card">
    <p style="color:var(--text2);margin-bottom:16px">Define the visual identity and brand voice for AI-generated content.</p>
    <form method="POST" action="/character/save" enctype="multipart/form-data">
        <div class="form-row">
            <label>Reference Image</label>
            {% if config.get('reference_image') %}
            <div style="margin-bottom:8px">
                <img src="/uploads/{{ config.get('reference_image') }}" style="max-width:200px;max-height:200px;border-radius:8px;border:1px solid var(--border)" onerror="this.style.display='none'">
                <div style="font-size:12px;color:var(--text2);margin-top:4px">Current: {{ config.get('reference_image') }}</div>
            </div>
            {% endif %}
            <input type="file" name="reference_image" accept="image/*">
            <span style="font-size:12px;color:var(--text2);display:block;margin-top:2px">Upload a reference photo. Use the physical description below to describe this person for image prompts.</span>
        </div>
        <div class="form-row">
            <label>Physical Description</label>
            <textarea name="physical_description" rows="3" style="width:100%" placeholder="Describe the person's appearance for image generation...">{{ config.get('physical_description', '') }}</textarea>
            <span style="font-size:12px;color:var(--text2);display:block;margin-top:2px">Used in image generation prompts to maintain character likeness</span>
        </div>
        <div class="form-row">
            <label>Style Descriptors</label>
            <textarea name="style_descriptors" rows="2" style="width:100%" placeholder="e.g. warm lighting, candid photography, lifestyle aesthetic...">{{ config.get('style_descriptors', '') }}</textarea>
        </div>
        <div class="form-row">
            <label>Camera Preferences</label>
            <textarea name="camera_preferences" rows="2" style="width:100%" placeholder="e.g. shot on Sony A7III, 85mm lens, shallow depth of field...">{{ config.get('camera_preferences', '') }}</textarea>
        </div>
        <div class="form-row">
            <label>Brand Voice</label>
            <textarea name="brand_voice" rows="3" style="width:100%" placeholder="Describe the tone and voice for captions...">{{ config.get('brand_voice', '') }}</textarea>
            <span style="font-size:12px;color:var(--text2);display:block;margin-top:2px">Used when generating Instagram captions</span>
        </div>
        <div class="form-row">
            <label>Default Hashtags</label>
            <textarea name="default_hashtags" rows="2" style="width:100%" placeholder="#hashtag1 #hashtag2 ...">{{ config.get('default_hashtags', '') }}</textarea>
        </div>
        <div class="form-row">
            <label>Infographic Style</label>
            <textarea name="infographic_style" rows="2" style="width:100%" placeholder="e.g. minimalist, muted tones, clean typography...">{{ config.get('infographic_style', '') }}</textarea>
        </div>
        <button class="btn btn-primary mt-8" type="submit">Save Character Config</button>
    </form>
</div>
"""
    return render_page(content, active="character", config=config)


@app.route("/character/save", methods=["POST"])
@login_required
def character_save():
    keys = ["physical_description", "style_descriptors", "camera_preferences",
            "brand_voice", "default_hashtags", "infographic_style"]
    db = get_db()
    for key in keys:
        val = request.form.get(key, "").strip()
        db.execute("INSERT OR REPLACE INTO character_config (key, value) VALUES (?, ?)", (key, val))

    # Handle reference image upload
    ref_file = request.files.get("reference_image")
    if ref_file and ref_file.filename:
        ref_filename = save_upload(ref_file)
        if ref_filename:
            # Delete old reference image
            old = db.execute("SELECT value FROM character_config WHERE key = 'reference_image'").fetchone()
            if old and old["value"]:
                delete_media(old["value"])
            db.execute("INSERT OR REPLACE INTO character_config (key, value) VALUES ('reference_image', ?)", (ref_filename,))

    db.commit()
    db.close()
    flash("Character config saved")
    return redirect("/character")


# ── Generated Content ─────────────────────────────────────────────────────────

@app.route("/generated")
@login_required
def generated():
    db = get_db()
    status_filter = request.args.get("status", "")
    page_num = max(1, int(request.args.get("page", 1)))
    per_page = 30

    where = ["1=1"]
    params = []
    if status_filter:
        where.append("gs.status = ?")
        params.append(status_filter)

    where_clause = " AND ".join(where)
    total = db.execute(f"SELECT COUNT(*) FROM generation_sessions gs WHERE {where_clause}", params).fetchone()[0]
    rows = db.execute(
        f"""SELECT gs.*, t.name as topic_name FROM generation_sessions gs
            LEFT JOIN topics t ON gs.topic_id = t.id
            WHERE {where_clause}
            ORDER BY gs.created_at DESC LIMIT ? OFFSET ?""",
        params + [per_page, (page_num - 1) * per_page],
    ).fetchall()
    db.close()

    total_pages = max(1, (total + per_page - 1) // per_page)

    content = """
<div class="flex-between">
    <div class="page-title">Generated Content</div>
    <a href="/generate/new" class="btn btn-primary">+ New Generation</a>
</div>

<div class="card section">
    <form method="GET" action="/generated">
        <div class="flex">
            <select name="status" style="width:200px">
                <option value="">All Statuses</option>
                <option value="questioning" {{ 'selected' if status_filter == 'questioning' }}>Questioning</option>
                <option value="awaiting_response" {{ 'selected' if status_filter == 'awaiting_response' }}>Awaiting Response</option>
                <option value="transcribing" {{ 'selected' if status_filter == 'transcribing' }}>Transcribing</option>
                <option value="generating" {{ 'selected' if status_filter == 'generating' }}>Generating</option>
                <option value="content_ready" {{ 'selected' if status_filter == 'content_ready' }}>Content Ready</option>
                <option value="inserted" {{ 'selected' if status_filter == 'inserted' }}>Inserted</option>
                <option value="failed" {{ 'selected' if status_filter == 'failed' }}>Failed</option>
                <option value="cancelled" {{ 'selected' if status_filter == 'cancelled' }}>Cancelled</option>
            </select>
            <button class="btn btn-primary" type="submit">Filter</button>
        </div>
    </form>
</div>

<div class="card section">
    <div class="flex-between mb-8">
        <h3 style="margin:0">{{ total }} session{{ 's' if total != 1 }}</h3>
        <div class="flex" style="font-size:13px;color:var(--text2)">
            Page {{ page_num }} of {{ total_pages }}
            {% if page_num > 1 %}<a href="?status={{ status_filter }}&page={{ page_num-1 }}">&laquo; Prev</a>{% endif %}
            {% if page_num < total_pages %}<a href="?status={{ status_filter }}&page={{ page_num+1 }}">Next &raquo;</a>{% endif %}
        </div>
    </div>

    {% if rows %}
    <table>
        <thead><tr><th>Date</th><th>Topic</th><th>Status</th><th>Media</th><th>Caption</th><th>Actions</th></tr></thead>
        <tbody>
        {% for s in rows %}
        <tr>
            <td style="white-space:nowrap">{{ ts_format(s.created_at) }}</td>
            <td>{{ s.topic_name or 'N/A' }}</td>
            <td>
                {% if s.status == 'content_ready' %}
                    <span class="badge badge-ok">Ready</span>
                {% elif s.status == 'inserted' %}
                    <span class="badge badge-info">Inserted</span>
                {% elif s.status == 'failed' %}
                    <span class="badge badge-err">Failed</span>
                {% elif s.status == 'cancelled' %}
                    <span class="badge badge-pending">Cancelled</span>
                {% elif s.status == 'generating' %}
                    <span class="badge badge-warn">Generating</span>
                {% else %}
                    <span class="badge badge-pending">{{ s.status }}</span>
                {% endif %}
            </td>
            <td>
                {% if s.generated_media_type %}
                    <span class="badge badge-info">{{ s.generated_media_type }}</span>
                {% else %}
                    -
                {% endif %}
            </td>
            <td style="max-width:250px;overflow:hidden;text-overflow:ellipsis">
                {{ s.generated_caption[:60] if s.generated_caption else '(none)' }}{% if s.generated_caption and s.generated_caption|length > 60 %}...{% endif %}
            </td>
            <td style="white-space:nowrap">
                <div class="flex">
                {% if s.status == 'content_ready' %}
                    <form method="POST" action="/generated/approve/{{ s.id }}">
                        <button class="btn btn-sm btn-success" type="submit">Schedule</button>
                    </form>
                    <form method="POST" action="/generated/post-now/{{ s.id }}">
                        <button class="btn btn-sm btn-primary" type="submit">Post Now</button>
                    </form>
                {% endif %}
                {% if s.generated_media_path %}
                    <a href="/uploads/{{ s.generated_media_path }}" target="_blank" class="btn btn-sm btn-primary">Preview</a>
                {% endif %}
                {% if s.status in ('content_ready', 'failed', 'cancelled') %}
                    <form method="POST" action="/generated/delete/{{ s.id }}" onsubmit="return confirm('Delete this session?')">
                        <button class="btn btn-sm btn-danger" type="submit">Delete</button>
                    </form>
                {% endif %}
                </div>
            </td>
        </tr>
        {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p style="color:var(--text2)">No generation sessions found.</p>
    {% endif %}
</div>
"""
    return render_page(
        content, active="generated",
        rows=rows, total=total, page_num=page_num, total_pages=total_pages,
        status_filter=status_filter, ts_format=ts_format,
    )


@app.route("/generated/approve/<int:session_id>", methods=["POST"])
@login_required
def generated_approve(session_id):
    db = get_db()
    sess = db.execute("SELECT * FROM generation_sessions WHERE id = ? AND status = 'content_ready'", (session_id,)).fetchone()
    if not sess:
        db.close()
        flash("Session not found or not ready", "error")
        return redirect("/generated")

    if not sess["generated_media_path"]:
        db.close()
        flash("No media generated for this session", "error")
        return redirect("/generated")

    # Determine post type
    media_type = sess["generated_media_type"] or "image"
    post_type = "reel" if media_type == "reel" else "image"

    # Find the next day with nothing scheduled (pending or posted)
    from datetime import datetime, timedelta
    occupied_days = set()
    all_scheduled = db.execute(
        "SELECT scheduled_at FROM posts WHERE status IN ('pending', 'posted') AND scheduled_at IS NOT NULL"
    ).fetchall()
    for row in all_scheduled:
        day = datetime.fromtimestamp(row["scheduled_at"]).date()
        occupied_days.add(day)
    # Start from tomorrow, find first open day, schedule at 10am local
    candidate = datetime.now().date() + timedelta(days=1)
    for _ in range(365):
        if candidate not in occupied_days:
            break
        candidate += timedelta(days=1)
    scheduled_at = datetime.combine(candidate, datetime.min.time().replace(hour=10)).timestamp()

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
        "INSERT INTO post_log (post_id, action, details, timestamp) VALUES (?, 'queued', 'Added from generated content', ?)",
        (post_id, time.time()),
    )
    db.commit()
    db.close()
    flash(f"Post #{post_id} scheduled for {candidate.strftime('%a %b %d')} at 10:00 AM")
    return redirect("/generated")


@app.route("/generated/post-now/<int:session_id>", methods=["POST"])
@login_required
def generated_post_now(session_id):
    db = get_db()
    sess = db.execute("SELECT * FROM generation_sessions WHERE id = ? AND status = 'content_ready'", (session_id,)).fetchone()
    if not sess:
        db.close()
        flash("Session not found or not ready", "error")
        return redirect("/generated")

    if not sess["generated_media_path"]:
        db.close()
        flash("No media generated for this session", "error")
        return redirect("/generated")

    media_type = sess["generated_media_type"] or "image"
    post_type = "reel" if media_type == "reel" else "image"

    db.execute(
        """INSERT INTO posts (post_type, media_path, caption, hashtags, status, scheduled_at,
           sort_order, share_to_story, generation_session_id, created_at)
           VALUES (?, ?, ?, ?, 'pending', ?, 0, 1, ?, ?)""",
        (post_type, sess["generated_media_path"], sess["generated_caption"] or "",
         sess["generated_hashtags"] or "", time.time() - 1, session_id, time.time()),
    )
    post_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    db.execute(
        "UPDATE generation_sessions SET status = 'inserted', post_id = ?, updated_at = ? WHERE id = ?",
        (post_id, time.time(), session_id),
    )
    db.execute(
        "INSERT INTO post_log (post_id, action, details, timestamp) VALUES (?, 'queued', 'Post now from generated content', ?)",
        (post_id, time.time()),
    )
    db.commit()
    db.close()
    flash(f"Post #{post_id} scheduled for immediate posting")
    return redirect("/generated")


@app.route("/generated/delete/<int:session_id>", methods=["POST"])
@login_required
def generated_delete(session_id):
    db = get_db()
    sess = db.execute("SELECT generated_media_path FROM generation_sessions WHERE id = ?", (session_id,)).fetchone()
    if sess:
        delete_media(sess["generated_media_path"])
        db.execute("DELETE FROM generation_sessions WHERE id = ?", (session_id,))
        db.commit()
        flash("Session deleted")
    db.close()
    return redirect("/generated")


# ── Generate (dashboard-initiated) ────────────────────────────────────────────

@app.route("/generate/new", methods=["GET"])
@login_required
def generate_new():
    db = get_db()
    topics_list = db.execute("SELECT * FROM topics WHERE enabled = 1 ORDER BY sort_order ASC, id ASC").fetchall()
    db.close()
    selected_topic_id = request.args.get("topic_id", "")

    content = """
<div class="page-title">New Generation</div>

<div class="card">
    <form method="POST" action="/generate/new">
        <div class="form-row">
            <label>Topic</label>
            <select name="topic_id" style="width:400px" required>
                <option value="">Select a topic...</option>
                {% for t in topics_list %}
                <option value="{{ t.id }}" {{ 'selected' if t.id|string == selected_topic_id }}>{{ t.name }}</option>
                {% endfor %}
            </select>
        </div>

        <div class="form-row">
            <label>Your Response / Content Input</label>
            <textarea name="response_text" rows="5" style="width:100%" required placeholder="Write your thoughts, story, or talking points for this post..."></textarea>
            <span style="font-size:12px;color:var(--text2);display:block;margin-top:2px">This will be used to generate the caption and image</span>
        </div>

        <div class="form-row">
            <label>Media Type</label>
            <select name="media_type" style="width:200px">
                <option value="image">Image</option>
                <option value="infographic">Infographic</option>
                <option value="reel">Reel (Video)</option>
            </select>
        </div>

        <div class="form-row">
            <label style="display:inline">
                <input type="checkbox" name="auto_queue" value="1" style="width:auto">
                Auto-queue after generation
            </label>
            <span style="font-size:12px;color:var(--text2);display:block;margin-top:2px">Automatically add to posting queue once content is ready</span>
        </div>

        <button class="btn btn-primary mt-8" type="submit">Generate Content</button>
    </form>
</div>
"""
    return render_page(content, active="generated", topics_list=topics_list, selected_topic_id=str(selected_topic_id))


@app.route("/generate/new", methods=["POST"])
@login_required
def generate_new_submit():
    topic_id = request.form.get("topic_id", "")
    response_text = request.form.get("response_text", "").strip()
    media_type = request.form.get("media_type", "image")
    auto_queue = "1" if request.form.get("auto_queue") else "0"

    if not topic_id or not response_text:
        flash("Topic and response are required", "error")
        return redirect("/generate/new")

    now = time.time()
    db = get_db()
    topic = db.execute("SELECT name FROM topics WHERE id = ?", (int(topic_id),)).fetchone()
    db.execute(
        """INSERT INTO generation_sessions
           (topic_id, status, question_text, response_text, response_type,
            generated_media_type, schedule_mode, created_at, updated_at)
           VALUES (?, 'pending_generation', ?, ?, 'dashboard', ?, ?, ?, ?)""",
        (int(topic_id), f"Dashboard generation for {topic['name'] if topic else 'topic'}", response_text,
         media_type, 'auto_queue' if auto_queue == '1' else 'queue', now, now),
    )
    session_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    db.commit()
    db.close()

    flash(f"Generation session #{session_id} created. The generator bot will process it shortly.")
    return redirect("/generated")


# ── Settings ───────────────────────────────────────────────────────────────────

@app.route("/settings")
@login_required
def settings():
    env = read_env()
    default_hashtags = get_setting("default_hashtags")
    auto_share_story = get_setting("auto_share_story", "1")
    story_reshare_enabled = get_setting("story_reshare_enabled", "0")
    story_reshare_count = get_setting("story_reshare_count", "5")
    generator_enabled = get_setting("generator_enabled", "0")
    generator_question_interval_hours = get_setting("generator_question_interval_hours", "4")
    generator_default_media_type = get_setting("generator_default_media_type", "image")
    generator_auto_post = get_setting("generator_auto_post", "0")
    generator_telegram_chat_id = get_setting("generator_telegram_chat_id", "")
    poster_status = get_launchd_status(POSTER_SERVICE)
    poster_pids = is_process_running("poster.py")
    generator_pids = is_process_running("generator.py")
    poster_log = tail_file(POSTER_LOG, 100)

    content = """
<div class="page-title">Settings</div>

<div class="card section">
    <h3>Instagram Credentials</h3>
    <form method="POST" action="/settings/save">
        <div class="form-row">
            <label>Username</label>
            <input type="text" name="INSTAGRAM_USERNAME" value="{{ env.get('INSTAGRAM_USERNAME', '') }}" style="width:400px">
        </div>
        <div class="form-row">
            <label>Password</label>
            <input type="password" name="INSTAGRAM_PASSWORD" value="{{ env.get('INSTAGRAM_PASSWORD', '') }}" style="width:400px">
        </div>
</div>

<div class="card section">
    <h3>Posting</h3>
        <div class="form-row">
            <label>Default Hashtags</label>
            <textarea name="default_hashtags" rows="2" style="width:400px">{{ default_hashtags }}</textarea>
        </div>
        <div class="form-row">
            <label>Check Interval (seconds)</label>
            <input type="number" name="CHECK_INTERVAL" value="{{ env.get('CHECK_INTERVAL', '60') }}" style="width:200px" min="10">
        </div>
        <div class="form-row">
            <label>Max Retries</label>
            <input type="number" name="MAX_RETRIES" value="{{ env.get('MAX_RETRIES', '3') }}" style="width:200px" min="0" max="10">
        </div>
        <div class="form-row">
            <label style="display:inline">
                <input type="checkbox" name="auto_share_story" value="1" {{ 'checked' if auto_share_story == '1' }} style="width:auto">
                Auto-share new posts to story
            </label>
            <span style="font-size:12px;color:var(--text2);display:block;margin-top:2px">When enabled, image and reel posts are automatically shared to your story after publishing</span>
        </div>
</div>

<div class="card section">
    <h3>Feed-to-Story Resharing</h3>
        <div class="form-row">
            <label style="display:inline">
                <input type="checkbox" name="story_reshare_enabled" value="1" {{ 'checked' if story_reshare_enabled == '1' }} style="width:auto">
                Enable automatic feed resharing
            </label>
            <span style="font-size:12px;color:var(--text2);display:block;margin-top:2px">Randomly picks posts from your feed and shares them to your story throughout the day</span>
        </div>
        <div class="form-row">
            <label>Times per day</label>
            <input type="number" name="story_reshare_count" value="{{ story_reshare_count }}" style="width:200px" min="1" max="24">
            <span style="font-size:12px;color:var(--text2);display:block;margin-top:2px">Posts will be spread evenly — e.g. 5x/day = roughly every 5 hours</span>
        </div>

        <button class="btn btn-primary mt-8" type="submit">Save Settings</button>
    </form>
</div>

<div class="card section">
    <h3>Content Generator</h3>
    <form method="POST" action="/settings/save-generator">
        <div class="form-row">
            <label style="display:inline">
                <input type="checkbox" name="generator_enabled" value="1" {{ 'checked' if generator_enabled == '1' }} style="width:auto">
                Generator Enabled
            </label>
            <span style="font-size:12px;color:var(--text2);display:block;margin-top:2px">Enable the AI content generation pipeline via Telegram bot</span>
        </div>
        <div class="form-row">
            <label>Question Interval (hours)</label>
            <input type="number" name="generator_question_interval_hours" value="{{ generator_question_interval_hours }}" style="width:200px" min="1" max="168" step="1">
            <span style="font-size:12px;color:var(--text2);display:block;margin-top:2px">How often the bot sends a topic question</span>
        </div>
        <div class="form-row">
            <label>Default Media Type</label>
            <select name="generator_default_media_type" style="width:200px">
                <option value="image" {{ 'selected' if generator_default_media_type == 'image' }}>Image</option>
                <option value="reel" {{ 'selected' if generator_default_media_type == 'reel' }}>Reel</option>
                <option value="infographic" {{ 'selected' if generator_default_media_type == 'infographic' }}>Infographic</option>
            </select>
        </div>
        <div class="form-row">
            <label style="display:inline">
                <input type="checkbox" name="generator_auto_post" value="1" {{ 'checked' if generator_auto_post == '1' }} style="width:auto">
                Auto-post generated content
            </label>
            <span style="font-size:12px;color:var(--text2);display:block;margin-top:2px">Automatically queue generated content without manual approval</span>
        </div>
        <div class="form-row">
            <label>Owner Telegram Chat ID</label>
            <input type="text" name="generator_telegram_chat_id" value="{{ generator_telegram_chat_id }}" style="width:300px" placeholder="e.g. 123456789">
            <span style="font-size:12px;color:var(--text2);display:block;margin-top:2px">Your Telegram user ID for receiving questions</span>
        </div>
        <button class="btn btn-primary mt-8" type="submit">Save Generator Settings</button>
    </form>
</div>

<div class="card section">
    <h3>Services</h3>
    <table class="mb-8">
        <tr>
            <td style="width:200px">Poster Daemon</td>
            <td>
                {% if poster_status.running %}
                    <span class="badge badge-ok">Running</span> PID {{ poster_status.pid }}
                {% elif poster_pids %}
                    <span class="badge badge-ok">Running</span> PID {{ poster_pids[0] }} (manual)
                {% else %}
                    <span class="badge badge-err">Stopped</span>
                {% endif %}
            </td>
        </tr>
        <tr>
            <td>Generator Bot</td>
            <td>
                {% if generator_pids %}
                    <span class="badge badge-ok">Running</span> PID {{ generator_pids[0] }}
                {% else %}
                    <span class="badge badge-err">Stopped</span>
                {% endif %}
            </td>
        </tr>
    </table>
    <div class="flex">
        <form method="POST" action="/service/restart"><button class="btn btn-warn" type="submit">Restart</button></form>
        <form method="POST" action="/service/stop"><button class="btn btn-danger" type="submit">Stop</button></form>
        <form method="POST" action="/service/start"><button class="btn btn-primary" type="submit">Start</button></form>
    </div>
</div>

<div class="card section">
    <h3>Poster Log (last 100 lines)</h3>
    <div class="log-viewer">{{ poster_log }}</div>
</div>
"""
    return render_page(
        content, active="settings",
        env=env, default_hashtags=default_hashtags,
        auto_share_story=auto_share_story,
        story_reshare_enabled=story_reshare_enabled,
        story_reshare_count=story_reshare_count,
        generator_enabled=generator_enabled,
        generator_question_interval_hours=generator_question_interval_hours,
        generator_default_media_type=generator_default_media_type,
        generator_auto_post=generator_auto_post,
        generator_telegram_chat_id=generator_telegram_chat_id,
        poster_status=poster_status, poster_pids=poster_pids,
        generator_pids=generator_pids,
        poster_log=poster_log,
    )


@app.route("/settings/save", methods=["POST"])
@login_required
def settings_save():
    updates = {}
    for key in EDITABLE_KEYS:
        val = request.form.get(key)
        if val is not None:
            updates[key] = val
    if updates:
        write_env(updates)

    default_hashtags = request.form.get("default_hashtags", "").strip()
    set_setting("default_hashtags", default_hashtags)
    auto_share_story = "1" if request.form.get("auto_share_story") else "0"
    set_setting("auto_share_story", auto_share_story)
    story_reshare_enabled = "1" if request.form.get("story_reshare_enabled") else "0"
    set_setting("story_reshare_enabled", story_reshare_enabled)
    story_reshare_count = request.form.get("story_reshare_count", "5").strip()
    set_setting("story_reshare_count", story_reshare_count)

    try:
        subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{POSTER_SERVICE}"],
            capture_output=True, timeout=10,
        )
        flash("Settings saved. Poster restarting...")
    except Exception:
        try:
            subprocess.run(["launchctl", "stop", POSTER_SERVICE], capture_output=True, timeout=5)
            time.sleep(1)
            subprocess.run(["launchctl", "start", POSTER_SERVICE], capture_output=True, timeout=5)
            flash("Settings saved. Poster restarting...")
        except Exception:
            flash("Settings saved. Restart poster manually to apply.")

    return redirect("/settings")


@app.route("/settings/save-generator", methods=["POST"])
@login_required
def settings_save_generator():
    generator_enabled = "1" if request.form.get("generator_enabled") else "0"
    set_setting("generator_enabled", generator_enabled)
    set_setting("generator_question_interval_hours", request.form.get("generator_question_interval_hours", "4").strip())
    set_setting("generator_default_media_type", request.form.get("generator_default_media_type", "image").strip())
    generator_auto_post = "1" if request.form.get("generator_auto_post") else "0"
    set_setting("generator_auto_post", generator_auto_post)
    set_setting("generator_telegram_chat_id", request.form.get("generator_telegram_chat_id", "").strip())
    flash("Generator settings saved")
    return redirect("/settings")


# ── Service management ─────────────────────────────────────────────────────────

@app.route("/service/restart", methods=["POST"])
@login_required
def service_restart():
    try:
        subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{POSTER_SERVICE}"],
            capture_output=True, timeout=10,
        )
        flash("Poster restart triggered")
    except Exception:
        try:
            subprocess.run(["launchctl", "stop", POSTER_SERVICE], capture_output=True, timeout=5)
            time.sleep(1)
            subprocess.run(["launchctl", "start", POSTER_SERVICE], capture_output=True, timeout=5)
            flash("Poster restarted (stop/start)")
        except Exception as e:
            flash(f"Error restarting poster: {e}", "error")
    return redirect(request.referrer or "/")


@app.route("/service/stop", methods=["POST"])
@login_required
def service_stop():
    try:
        subprocess.run(["launchctl", "stop", POSTER_SERVICE], capture_output=True, timeout=5)
        flash("Poster stopped")
    except Exception as e:
        flash(f"Error: {e}", "error")
    return redirect("/settings")


@app.route("/service/start", methods=["POST"])
@login_required
def service_start():
    try:
        subprocess.run(["launchctl", "start", POSTER_SERVICE], capture_output=True, timeout=5)
        flash("Poster started")
    except Exception as e:
        flash(f"Error: {e}", "error")
    return redirect("/settings")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"IG Autoposter Dashboard running on http://localhost:{PORT}")
    print(f"Password: {'(set via DASHBOARD_PASSWORD)' if os.getenv('DASHBOARD_PASSWORD') else DASHBOARD_PASSWORD}")
    app.run(host="0.0.0.0", port=PORT, debug=True)
