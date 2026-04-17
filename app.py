"""
============================================================
  ATTENTUS — Servidor Central de Coleta de Dados
  GenMate Field Intelligence Platform
============================================================
  Recebe dados de estações meteorológicas (ESP32+DHT22+BMP280+BH1750)
  e monitor de bezerras (ESP32-CAM) via HTTP.

  Endpoints de ingestão:
    POST /api/sensors   → JSON da estação meteorológica
    POST /api/upload    → multipart/form-data do calf monitor
    POST /api/perspicuus/events → JSON (application/json) ou multipart com campo "json" + ficheiros frontal_1, lateral_1, …

  Implantação: Render.com
    - Web Service (Python / Gunicorn)
    - Persistent Disk em /data (SQLite + imagens)

  Variáveis de ambiente:
    SECRET_KEY    → chave Flask (obrigatório em prod)
    ADMIN_USER    → usuário admin (padrão: admin)
    ADMIN_PASS    → senha admin (padrão: attentus2024)
    API_KEY       → chave dos dispositivos (opcional; vazio = sem auth)
    DATA_DIR      → diretório de dados (padrão: /data no Render, ./data local)
============================================================
"""

import os
import re
import sqlite3
import json
import threading
import zipfile
import io
import csv
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import wraps
from zoneinfo import ZoneInfo
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, send_file, flash, g, abort
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from thi_calculator import calculate_thi, thi_holstein_zones_for_chart, thi_thresholds_json

TZ_BR = ZoneInfo('America/Sao_Paulo')


def _parse_utc_received(iso_str):
    """Interpreta timestamps ISO gravados em UTC (sufixo Z ou offset)."""
    if not iso_str:
        return None
    s = str(iso_str).strip()
    try:
        if s.endswith('Z'):
            return datetime.fromisoformat(s.replace('Z', '+00:00'))
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def format_br_datetime(iso_str, fmt='%d/%m/%Y %H:%M'):
    dt = _parse_utc_received(iso_str)
    if dt is None:
        return '—'
    return dt.astimezone(TZ_BR).strftime(fmt)


# ─── CONFIG ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-CHANGE-IN-PRODUCTION')


@app.template_filter('br_dt')
def template_br_dt(value, fmt='%d/%m/%Y %H:%M'):
    return format_br_datetime(value, fmt) if value else '—'

# Diretório de dados: /data no Render (disco persistente), ./data local
DATA_DIR = os.environ.get(
    'DATA_DIR',
    '/data' if os.environ.get('RENDER') else os.path.join(os.path.dirname(__file__), 'data')
)
DB_PATH     = os.path.join(DATA_DIR, 'attentus.db')
UPLOADS_DIR = os.path.join(DATA_DIR, 'uploads')
os.makedirs(UPLOADS_DIR, exist_ok=True)
# Alinha perspicuus_inference.get_models_dir() com o mesmo DATA_DIR do processo
os.environ.setdefault('DATA_DIR', DATA_DIR)
ML_MODELS_DIR = os.path.join(DATA_DIR, 'ml_models')
os.makedirs(ML_MODELS_DIR, exist_ok=True)
MAX_MODEL_UPLOAD_BYTES = int(os.environ.get('MAX_MODEL_UPLOAD_MB', '500')) * 1024 * 1024
PERSPICUUS_MODEL_ROLE_EXT = {
    'yolo': {'.onnx'},
    'lateral': {'.onnx'},
    'posterior': {'.onnx'},
    'lateral_meta': {'.json'},
    'posterior_meta': {'.json'},
}
log.info(f"DATA_DIR={DATA_DIR}")

# Autenticação de usuário web
ADMIN_USER      = os.environ.get('ADMIN_USER', 'admin')
ADMIN_PASS_HASH = generate_password_hash(os.environ.get('ADMIN_PASS', 'attentus2024'))

# Chave de API para dispositivos (vazio = sem restrição)
API_KEY = os.environ.get('API_KEY', '')
PERSPICUUS_IMAGE_KEYS = ('frontal', 'lateral', 'posterior', 'superior')
PERSPICUUS_MULTIPART_FIELD = re.compile(
    r'^(frontal|lateral|posterior|superior)_(\d+)$',
    re.IGNORECASE,
)
PERSPICUUS_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


def _perspicuus_media_url_from_path(path):
    """URL utilizável em <img src> para ficheiros deste servidor ou URLs absolutas."""
    if not path:
        return None
    p = str(path).strip()
    if p.startswith('/api/perspicuus/media/'):
        return p
    if p.startswith(('http://', 'https://')):
        return p
    return None


@app.template_filter('perspicuus_media_url')
def template_perspicuus_media_url(path):
    u = _perspicuus_media_url_from_path(path)
    return u if u else ''


@app.template_filter('perspicuus_first_preview')
def template_perspicuus_first_preview(row):
    """Primeira imagem servível a partir de uma linha `perspicuus_events`."""
    d = row if isinstance(row, dict) else dict(row)
    for view in PERSPICUUS_IMAGE_KEYS:
        raw = d.get(f'{view}_json') or '[]'
        try:
            frames = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            u = _perspicuus_media_url_from_path(frame.get('path'))
            if u:
                return u
    return ''


# Temperatura/umidade: DHT22 (dht22_*) ou chaves legadas bmp280/bme280 no JSON do firmware
def _ingest_temp_c(data):
    if not isinstance(data, dict):
        return None
    for key in ('dht22_temp_c', 'bmp280_temp_c', 'bme280_temp_c'):
        if key not in data:
            continue
        v = _float_or_none(data[key])
        if v is not None:
            return v
    return None


def _ingest_humidity(data):
    if not isinstance(data, dict):
        return None
    for key in ('dht22_humidity', 'bme280_humidity', 'humidity'):
        if key not in data:
            continue
        v = _float_or_none(data[key])
        if v is not None:
            return v
    return None


def _float_or_none(v):
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode('utf-8', errors='ignore')
    if isinstance(v, str) and not v.strip():
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _temp_humidity_for_thi(row_dict):
    """
    Temperatura e UR para cálculo do THI: colunas SQLite; se faltar algum valor,
    reutiliza as mesmas chaves do POST em raw_json (dados legados ou colunas vazias).
    """
    t = _float_or_none(row_dict.get('temp_c'))
    h = _float_or_none(row_dict.get('humidity'))
    if t is not None and h is not None:
        return t, h
    raw = row_dict.get('raw_json')
    if not raw:
        return t, h
    try:
        j = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return t, h
    if not isinstance(j, dict):
        return t, h
    if t is None:
        t = _float_or_none(_ingest_temp_c(j))
    if h is None:
        h = _float_or_none(_ingest_humidity(j))
    return t, h


def _sanitize_perspicuus_frames(items):
    """Normaliza lista de frames: [{'frame_index': int, 'path': str}, ...]."""
    out = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        path = str(item.get('path', '')).strip()
        if not path:
            continue
        try:
            frame_index = int(item.get('frame_index', len(out) + 1))
        except (TypeError, ValueError):
            frame_index = len(out) + 1
        out.append({'frame_index': frame_index, 'path': path})
    return out


def _serialize_perspicuus_row(row):
    """Converte sqlite Row em dict para API/UI (images como listas, payload parseado)."""
    d = dict(row)
    images = {}
    for k in PERSPICUUS_IMAGE_KEYS:
        raw = d.pop(f'{k}_json', '[]')
        try:
            images[k] = json.loads(raw) if raw else []
        except (json.JSONDecodeError, TypeError):
            images[k] = []
    d['images'] = images
    d['inference_ready'] = bool(d.get('inference_ready', 0))
    raw_s = d.get('raw_json') or ''
    try:
        d['payload'] = json.loads(raw_s) if raw_s else {}
    except (json.JSONDecodeError, TypeError):
        d['payload'] = {}
    inf_s = d.get('inference_json') or '{}'
    try:
        d['inference'] = json.loads(inf_s) if inf_s else {}
    except (json.JSONDecodeError, TypeError):
        d['inference'] = {}
    return d


def _perspicuus_image_ext(original_name):
    ext = os.path.splitext(original_name or '')[1].lower()
    return ext if ext in PERSPICUUS_EXTS else '.jpg'


def _merge_perspicuus_uploads(payload, files_storage, event_id):
    """
    Grava ficheiros multipart nomeados frontal_1, lateral_2, … em
    uploads/perspicuus/<event_id>/ e substitui payload['images'] pelos paths servidos.
    Retorna o número de ficheiros gravados (0 se nenhum campo reconhecido).
    """
    grouped = defaultdict(list)
    for key, fs in files_storage.items():
        if not fs or not getattr(fs, 'filename', None):
            continue
        m = PERSPICUUS_MULTIPART_FIELD.match(key.strip())
        if not m:
            continue
        view = m.group(1).lower()
        try:
            frame_idx = int(m.group(2))
        except ValueError:
            continue
        if view not in PERSPICUUS_IMAGE_KEYS:
            continue
        grouped[view].append((frame_idx, fs))

    if not grouped:
        return 0

    safe_dir = secure_filename(event_id) or 'event'
    dest_root = os.path.join(UPLOADS_DIR, 'perspicuus', safe_dir)
    os.makedirs(dest_root, exist_ok=True)

    images_out = {k: [] for k in PERSPICUUS_IMAGE_KEYS}
    n_saved = 0
    for view in PERSPICUUS_IMAGE_KEYS:
        for frame_idx, fs in sorted(grouped.get(view, []), key=lambda x: x[0]):
            ext = _perspicuus_image_ext(fs.filename)
            fname = f'{view}_{frame_idx}{ext}'
            dest = os.path.join(dest_root, fname)
            fs.save(dest)
            n_saved += 1
            url_path = f'/api/perspicuus/media/{safe_dir}/{fname}'
            images_out[view].append({'frame_index': frame_idx, 'path': url_path})

    payload['images'] = images_out
    payload['inference_ready'] = True
    return n_saved


def _normalize_perspicuus_payload(payload):
    """Valida e normaliza payload do brete inteligente."""
    if not isinstance(payload, dict):
        return None, 'JSON inválido'

    event_id = str(payload.get('event_id', '')).strip()
    timestamp_utc = str(payload.get('timestamp_utc', '')).strip()
    station_id = str(payload.get('station_id', '')).strip()
    device_id = str(payload.get('device_id', '')).strip()
    animal = payload.get('animal') if isinstance(payload.get('animal'), dict) else {}
    images = payload.get('images') if isinstance(payload.get('images'), dict) else {}

    animal_rfid = str(animal.get('rfid', '')).strip()
    animal_status = str(animal.get('status', '')).strip()
    try:
        animal_repetition = int(animal.get('repetition', 0))
    except (TypeError, ValueError):
        animal_repetition = 0

    if not event_id:
        return None, 'Campo obrigatório ausente: event_id'
    if not timestamp_utc or _parse_utc_received(timestamp_utc) is None:
        return None, 'timestamp_utc inválido (use ISO UTC, ex: 2026-04-15T17:33:55.182Z)'
    if not station_id:
        return None, 'Campo obrigatório ausente: station_id'
    if not device_id:
        return None, 'Campo obrigatório ausente: device_id'
    if not animal_rfid:
        return None, 'Campo obrigatório ausente: animal.rfid'

    by_view = {k: _sanitize_perspicuus_frames(images.get(k, [])) for k in PERSPICUUS_IMAGE_KEYS}
    total_images = sum(len(by_view[k]) for k in PERSPICUUS_IMAGE_KEYS)

    return {
        'event_id': event_id,
        'timestamp_utc': timestamp_utc,
        'station_id': station_id,
        'device_id': device_id,
        'animal_rfid': animal_rfid,
        'animal_status': animal_status,
        'animal_repetition': animal_repetition,
        'inference_ready': 1 if bool(payload.get('inference_ready', False)) else 0,
        'images': by_view,
        'total_images': total_images,
    }, None


ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
# Werkzeug rejeita o corpo antes da view; tem de ser >= uploads de modelos ONNX (MAX_MODEL_UPLOAD_MB).
MAX_CONTENT_LENGTH = max(10 * 1024 * 1024, MAX_MODEL_UPLOAD_BYTES)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# ─── BANCO DE DADOS ───────────────────────────────────────────────────────────

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db:
        db.close()

def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS weather (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        received_at TEXT    NOT NULL,
        device_name TEXT    NOT NULL DEFAULT 'unknown',
        lux         REAL,
        temp_c      REAL,
        press_hpa   REAL,
        alt_m       REAL,
        humidity    REAL,
        uptime_s    INTEGER,
        rssi        INTEGER,
        raw_json    TEXT
    );

    CREATE TABLE IF NOT EXISTS images (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        received_at TEXT    NOT NULL,
        device_name TEXT    NOT NULL DEFAULT 'unknown',
        capture_id  INTEGER,
        filename    TEXT,
        filesize    INTEGER,
        rssi        INTEGER,
        notes       TEXT    DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS perspicuus_events (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id          TEXT    NOT NULL UNIQUE,
        received_at       TEXT    NOT NULL,
        timestamp_utc     TEXT    NOT NULL,
        station_id        TEXT    NOT NULL,
        device_id         TEXT    NOT NULL,
        animal_rfid       TEXT    NOT NULL,
        animal_status     TEXT    DEFAULT '',
        animal_repetition INTEGER DEFAULT 0,
        inference_ready   INTEGER NOT NULL DEFAULT 0,
        frontal_json      TEXT    NOT NULL DEFAULT '[]',
        lateral_json      TEXT    NOT NULL DEFAULT '[]',
        posterior_json    TEXT    NOT NULL DEFAULT '[]',
        superior_json     TEXT    NOT NULL DEFAULT '[]',
        total_images      INTEGER NOT NULL DEFAULT 0,
        raw_json          TEXT    NOT NULL,
        inference_json    TEXT    NOT NULL DEFAULT '{}',
        inference_at      TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_weather_received ON weather(received_at);
    CREATE INDEX IF NOT EXISTS idx_weather_device   ON weather(device_name);
    CREATE INDEX IF NOT EXISTS idx_images_received  ON images(received_at);
    CREATE INDEX IF NOT EXISTS idx_images_device    ON images(device_name);
    CREATE INDEX IF NOT EXISTS idx_persp_ts         ON perspicuus_events(timestamp_utc);
    CREATE INDEX IF NOT EXISTS idx_persp_station    ON perspicuus_events(station_id);
    CREATE INDEX IF NOT EXISTS idx_persp_rfid       ON perspicuus_events(animal_rfid);
    """)
    db.commit()
    try:
        cols = {row[1] for row in db.execute("PRAGMA table_info(perspicuus_events)")}
        if "inference_json" not in cols:
            db.execute(
                "ALTER TABLE perspicuus_events ADD COLUMN inference_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "inference_at" not in cols:
            db.execute("ALTER TABLE perspicuus_events ADD COLUMN inference_at TEXT")
        db.commit()
    except sqlite3.OperationalError as e:
        log.warning("Migração perspicuus_events: %s", e)
    db.close()
    log.info("DB inicializado OK")

init_db()


def _perspicuus_inference_engine_ready():
    try:
        from perspicuus_inference import get_engine
        return get_engine().is_ready()
    except Exception:
        return False


def _perspicuus_auto_infer_enabled():
    v = os.environ.get('PERSPICUUS_AUTO_INFER', '1').strip().lower()
    return v not in ('0', 'false', 'no', 'off')


def _run_perspicuus_inference_job(row_id: int, trigger: str):
    """
    Background: processa cada frame em sequência; médias por vista em run_inference_for_event.
    Usa sqlite3 direto (sem Flask g).
    """
    try:
        from perspicuus_inference import run_inference_for_event, get_engine
        if not get_engine().is_ready():
            log.info('[PERSPICUUS] auto-infer ignorada (ONNX não configurado) id=%s', row_id)
            return
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            'SELECT * FROM perspicuus_events WHERE id = ?', (row_id,)
        ).fetchone()
        conn.close()
        if not row:
            return
        d = dict(row)
        result = run_inference_for_event(d, UPLOADS_DIR)
        result['trigger'] = trigger
        now = datetime.utcnow().isoformat() + 'Z'
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            'UPDATE perspicuus_events SET inference_json = ?, inference_at = ? WHERE id = ?',
            (json.dumps(result, ensure_ascii=False), now, row_id),
        )
        conn.commit()
        conn.close()
        log.info('[PERSPICUUS] inferência %s gravada id=%s', trigger, row_id)
    except Exception:
        log.exception('[PERSPICUUS] inferência em background falhou id=%s', row_id)


def _schedule_perspicuus_auto_infer(row_id: int):
    threading.Thread(
        target=_run_perspicuus_inference_job,
        args=(row_id, 'auto_ingest'),
        daemon=True,
    ).start()


# ─── DECORATORS ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated

def api_auth(f):
    """Autenticação por X-API-Key ou ?key= (dispensada se API_KEY não configurada)"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if API_KEY:
            key = request.headers.get('X-API-Key', '') or request.args.get('key', '')
            if key != API_KEY:
                log.warning(f"API auth failed from {request.remote_addr}")
                return jsonify({'error': 'Unauthorized', 'hint': 'Provide X-API-Key header'}), 401
        return f(*args, **kwargs)
    return decorated

# ─── AUTH ─────────────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        user = request.form.get('username', '').strip()
        pw   = request.form.get('password', '')
        if user == ADMIN_USER and check_password_hash(ADMIN_PASS_HASH, pw):
            session['logged_in'] = True
            session['username']  = user
            next_url = request.args.get('next', url_for('index'))
            return redirect(next_url)
        error = 'Usuário ou senha incorretos.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ─── PAGES ────────────────────────────────────────────────────────────────────

@app.route('/')
@login_required
def index():
    db = get_db()
    stats = {}
    stats['weather_total']  = db.execute("SELECT COUNT(*) FROM weather").fetchone()[0]
    stats['images_total']   = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    stats['cameras_active'] = db.execute("SELECT COUNT(DISTINCT device_name) FROM images").fetchone()[0]
    stats['stations_active']= db.execute("SELECT COUNT(DISTINCT device_name) FROM weather").fetchone()[0]

    r = db.execute("SELECT received_at FROM weather ORDER BY received_at DESC LIMIT 1").fetchone()
    stats['weather_last'] = format_br_datetime(r[0]) if r else '—'

    r = db.execute("SELECT received_at FROM images ORDER BY received_at DESC LIMIT 1").fetchone()
    stats['images_last'] = format_br_datetime(r[0]) if r else '—'

    recent_weather = db.execute(
        "SELECT * FROM weather ORDER BY received_at DESC LIMIT 8"
    ).fetchall()

    recent_images = db.execute(
        "SELECT * FROM images ORDER BY received_at DESC LIMIT 4"
    ).fetchall()

    # Atividade últimas 24h — agrupa por hora do relógio em Brasília
    since_utc = datetime.now(timezone.utc) - timedelta(hours=24)
    since_iso = since_utc.isoformat().replace('+00:00', 'Z')
    ar = db.execute(
        "SELECT received_at FROM weather WHERE received_at > ?",
        (since_iso,),
    ).fetchall()
    counts = defaultdict(int)
    for (received_at,) in ar:
        dt = _parse_utc_received(received_at)
        if dt:
            counts[dt.astimezone(TZ_BR).hour] += 1
    activity = [(f'{h:02d}', counts[h]) for h in range(24)]

    return render_template('index.html',
        stats=stats,
        recent_weather=recent_weather,
        recent_images=recent_images,
        activity=activity,
    )

@app.route('/weather')
@login_required
def weather():
    db = get_db()
    devices = [r[0] for r in db.execute(
        "SELECT DISTINCT device_name FROM weather ORDER BY device_name"
    ).fetchall()]
    return render_template(
        'weather.html',
        devices=devices,
        thi_zones=thi_holstein_zones_for_chart(),
        thi_thresholds=thi_thresholds_json(),
    )

@app.route('/calf-monitor')
@login_required
def calf_monitor():
    db = get_db()
    # Última imagem por monitor de bezerra
    rows = db.execute("""
        SELECT i.id, i.device_name, i.filename, i.received_at, i.filesize, i.rssi, i.capture_id, i.notes
        FROM images i
        INNER JOIN (
            SELECT device_name, MAX(received_at) as max_ts
            FROM images GROUP BY device_name
        ) latest ON i.device_name = latest.device_name
                  AND i.received_at = latest.max_ts
        ORDER BY i.received_at DESC
    """).fetchall()

    # Contagem total por monitor
    counts = dict(db.execute(
        "SELECT device_name, COUNT(*) FROM images GROUP BY device_name"
    ).fetchall())

    return render_template('cameras.html', cameras=rows, counts=counts)


@app.route('/perspicuus')
@login_required
def perspicuus():
    db = get_db()
    page = max(1, int(request.args.get('page', 1)))
    per_page = 20
    offset = (page - 1) * per_page
    station = request.args.get('station', '').strip()
    rfid = request.args.get('rfid', '').strip()
    ready = request.args.get('ready', '').strip()

    conditions, params = ['1=1'], []
    if station:
        conditions.append('station_id = ?')
        params.append(station)
    if rfid:
        conditions.append('animal_rfid LIKE ?')
        params.append(f'%{rfid}%')
    if ready in ('0', '1'):
        conditions.append('inference_ready = ?')
        params.append(int(ready))
    where = ' AND '.join(conditions)

    total = db.execute(f'SELECT COUNT(*) FROM perspicuus_events WHERE {where}', params).fetchone()[0]
    rows = db.execute(
        f"SELECT * FROM perspicuus_events WHERE {where} ORDER BY timestamp_utc DESC LIMIT ? OFFSET ?",
        params + [per_page, offset]
    ).fetchall()

    records = []
    for row in rows:
        d = dict(row)
        for key in PERSPICUUS_IMAGE_KEYS:
            raw = d.get(f'{key}_json') or '[]'
            try:
                d[key] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                d[key] = []
            d[f'{key}_count'] = len(d[key])
        d['inference_ready'] = bool(d.get('inference_ready', 0))
        inf_raw = d.get('inference_json') or '{}'
        try:
            d['inference'] = json.loads(inf_raw) if inf_raw else {}
        except (json.JSONDecodeError, TypeError):
            d['inference'] = {}
        records.append(d)

    total_pages = max(1, (total + per_page - 1) // per_page)
    stations = [r[0] for r in db.execute(
        'SELECT DISTINCT station_id FROM perspicuus_events ORDER BY station_id'
    ).fetchall()]
    stats = {
        'total': db.execute('SELECT COUNT(*) FROM perspicuus_events').fetchone()[0],
        'ready': db.execute('SELECT COUNT(*) FROM perspicuus_events WHERE inference_ready = 1').fetchone()[0],
        'stations': db.execute('SELECT COUNT(DISTINCT station_id) FROM perspicuus_events').fetchone()[0],
        'rfids': db.execute('SELECT COUNT(DISTINCT animal_rfid) FROM perspicuus_events').fetchone()[0],
    }

    return render_template(
        'perspicuus.html',
        records=records,
        stations=stations,
        station_filter=station,
        rfid_filter=rfid,
        ready_filter=ready,
        page=page,
        total_pages=total_pages,
        total=total,
        stats=stats,
        inference_engine_ready=_perspicuus_inference_engine_ready(),
    )


def _traits_mean_summary(inf: dict, view: str, max_traits: int = 5) -> str:
    """Uma linha curta para tabela (médias por trait)."""
    views = inf.get('views') or {}
    v = views.get(view)
    if isinstance(v, list):
        v = {'traits_mean': {}, 'frames': v}
    if not isinstance(v, dict):
        return '—'
    tm = v.get('traits_mean') or {}
    if not tm:
        return '—'
    items = list(tm.items())[:max_traits]
    s = ', '.join(f'{k}:{float(val):.2f}' for k, val in items)
    if len(tm) > max_traits:
        s += '…'
    return s


def _safe_load_json(text: str, fallback):
    try:
        return json.loads(text) if text else fallback
    except (json.JSONDecodeError, TypeError):
        return fallback


def _last_frame_path(raw_json: str) -> str | None:
    frames = _safe_load_json(raw_json or '[]', [])
    if not isinstance(frames, list) or not frames:
        return None
    for item in reversed(frames):
        if isinstance(item, dict):
            p = str(item.get('path', '')).strip()
            if p:
                return p
    return None


def _merged_traits_mean(inf: dict) -> dict[str, float]:
    views = inf.get('views') or {}
    merged: dict[str, list[float]] = defaultdict(list)
    for view in ('lateral', 'posterior'):
        block = views.get(view)
        if isinstance(block, list):
            block = {'traits_mean': {}, 'frames': block}
        if not isinstance(block, dict):
            continue
        tm = block.get('traits_mean') or {}
        if not isinstance(tm, dict):
            continue
        for k, v in tm.items():
            try:
                merged[str(k)].append(float(v))
            except (TypeError, ValueError):
                continue
    return {k: sum(vals) / len(vals) for k, vals in merged.items() if vals}


def _trait_sort_key(name: str):
    """Ordena T1, T2, … T10 corretamente; resto alfabético."""
    s = str(name)
    m = re.match(r'^T(\d+)$', s, re.IGNORECASE)
    if m:
        return (0, int(m.group(1)))
    return (1, s.lower())


def _parse_trait_order_env() -> list[str]:
    """
    Lista opcional de nomes de traits na ordem desejada (Animais MK1).
    Ex.: PERSPICUUS_TRAIT_ORDER=T1,T2,T3,ECC
    """
    raw = os.environ.get('PERSPICUUS_TRAIT_ORDER', '').strip()
    if not raw:
        return []
    out, seen = [], set()
    for part in raw.split(','):
        k = part.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _traits_detail_rows(
    latest: dict[str, float],
    hist_points: dict[str, list[float]],
) -> list[dict]:
    """
    Uma linha por trait, na sequência definida por PERSPICUUS_TRAIT_ORDER
    ou ordenação automática (T1… depois resto).
    """
    env_order = _parse_trait_order_env()
    all_keys = set(latest.keys()) | set(hist_points.keys())
    rows: list[dict] = []

    def _row(name: str) -> dict:
        cur = latest.get(name)
        vals = hist_points.get(name, [])
        return {
            'name': name,
            'current': float(cur) if cur is not None else None,
            'n': len(vals),
        }

    if env_order:
        for name in env_order:
            rows.append(_row(name))
        rest = sorted(all_keys - set(env_order), key=_trait_sort_key)
        for name in rest:
            rows.append(_row(name))
        return rows

    for name in sorted(all_keys, key=_trait_sort_key):
        rows.append(_row(name))
    return rows


@app.route('/perspicuus/animais')
@login_required
def perspicuus_animais():
    """Visão por animal: última inferência, histórico de traits e últimas fotos."""
    db = get_db()
    page = max(1, int(request.args.get('page', 1)))
    per_page = 12
    offset = (page - 1) * per_page
    station = request.args.get('station', '').strip()
    q = request.args.get('q', '').strip()

    conditions = ["animal_rfid IS NOT NULL", "trim(animal_rfid) != ''"]
    params: list[str] = []
    if station:
        conditions.append("station_id = ?")
        params.append(station)
    if q:
        like = f'%{q}%'
        conditions.append(
            "(animal_rfid LIKE ? OR animal_status LIKE ? OR station_id LIKE ?)"
        )
        params.extend([like, like, like])
    where = " AND ".join(conditions)

    total_animals = db.execute(
        f"SELECT COUNT(DISTINCT animal_rfid) FROM perspicuus_events WHERE {where}",
        params,
    ).fetchone()[0]
    total_pages = max(1, (total_animals + per_page - 1) // per_page) if total_animals else 1

    animals_rows = db.execute(
        f"""
        SELECT
            animal_rfid,
            MAX(timestamp_utc) AS last_ts,
            COUNT(*) AS n_events,
            SUM(CASE WHEN inference_at IS NOT NULL AND trim(inference_at) != '' THEN 1 ELSE 0 END) AS n_inferred
        FROM perspicuus_events
        WHERE {where}
        GROUP BY animal_rfid
        ORDER BY last_ts DESC
        LIMIT ? OFFSET ?
        """,
        params + [per_page, offset],
    ).fetchall()

    cards = []
    for r in animals_rows:
        rfid = str(r['animal_rfid'])
        last_row = db.execute(
            """
            SELECT *
            FROM perspicuus_events
            WHERE animal_rfid = ?
            ORDER BY timestamp_utc DESC
            LIMIT 1
            """,
            (rfid,),
        ).fetchone()
        if not last_row:
            continue
        latest = dict(last_row)
        latest_inf = _safe_load_json(latest.get('inference_json') or '{}', {})
        latest_traits = _merged_traits_mean(latest_inf)

        hist_rows = db.execute(
            """
            SELECT timestamp_utc, inference_json
            FROM perspicuus_events
            WHERE animal_rfid = ?
              AND inference_at IS NOT NULL
              AND trim(inference_at) != ''
            ORDER BY timestamp_utc DESC
            LIMIT 12
            """,
            (rfid,),
        ).fetchall()
        hist_points: dict[str, list[float]] = defaultdict(list)
        for hr in reversed(hist_rows):
            inf = _safe_load_json(hr['inference_json'] or '{}', {})
            for trait, val in _merged_traits_mean(inf).items():
                hist_points[trait].append(float(val))

        trait_cards = _traits_detail_rows(latest_traits, hist_points)

        cards.append({
            'rfid': rfid,
            'status': latest.get('animal_status') or '—',
            'station_id': latest.get('station_id') or '—',
            'device_id': latest.get('device_id') or '—',
            'last_ts': latest.get('timestamp_utc'),
            'last_event_id': latest.get('event_id'),
            'n_events': int(r['n_events'] or 0),
            'n_inferred': int(r['n_inferred'] or 0),
            'last_lateral_path': _last_frame_path(latest.get('lateral_json') or '[]'),
            'last_posterior_path': _last_frame_path(latest.get('posterior_json') or '[]'),
            'trait_cards': trait_cards,
        })

    stations = [row[0] for row in db.execute(
        'SELECT DISTINCT station_id FROM perspicuus_events ORDER BY station_id'
    ).fetchall()]
    stats = {
        'animals_total': db.execute(
            "SELECT COUNT(DISTINCT animal_rfid) FROM perspicuus_events WHERE animal_rfid IS NOT NULL AND trim(animal_rfid) != ''"
        ).fetchone()[0],
        'animals_with_inference': db.execute(
            """
            SELECT COUNT(DISTINCT animal_rfid)
            FROM perspicuus_events
            WHERE animal_rfid IS NOT NULL
              AND trim(animal_rfid) != ''
              AND inference_at IS NOT NULL
              AND trim(inference_at) != ''
            """
        ).fetchone()[0],
    }

    return render_template(
        'perspicuus_animais.html',
        cards=cards,
        page=page,
        total_pages=total_pages,
        total_animals=total_animals,
        station_filter=station,
        q_filter=q,
        stations=stations,
        stats=stats,
        trait_order_defined=bool(_parse_trait_order_env()),
    )


@app.route('/perspicuus/inferencias')
@login_required
def perspicuus_inferencias():
    """Gestão e visualização agregada das inferências MK1."""
    db = get_db()
    page = max(1, int(request.args.get('page', 1)))
    per_page = 25
    offset = (page - 1) * per_page
    station = request.args.get('station', '').strip()
    status_f = request.args.get('status', '').strip()
    q = request.args.get('q', '').strip()

    base_c, base_p = ['1=1'], []
    if station:
        base_c.append('station_id = ?')
        base_p.append(station)
    if q:
        like = f'%{q}%'
        base_c.append(
            '(event_id LIKE ? OR animal_rfid LIKE ? OR station_id LIKE ? OR inference_json LIKE ?)'
        )
        base_p.extend([like, like, like, like])
    base_where = ' AND '.join(base_c)

    list_c = list(base_c)
    list_p = list(base_p)
    if status_f == 'pending':
        list_c.append("(inference_at IS NULL OR trim(inference_at) = '')")
    elif status_f == 'ok':
        list_c.append("inference_at IS NOT NULL AND trim(inference_at) != ''")
        list_c.append('json_extract(inference_json, \'$.error\') IS NULL')
    elif status_f == 'error':
        list_c.append("json_extract(inference_json, '$.error') IS NOT NULL")

    where_list = ' AND '.join(list_c)

    def _safe_count(sql: str, p):
        try:
            return db.execute(sql, p).fetchone()[0]
        except sqlite3.OperationalError:
            return 0

    try:
        total = db.execute(
            f'SELECT COUNT(*) FROM perspicuus_events WHERE {where_list}', list_p
        ).fetchone()[0]
        rows = db.execute(
            f'SELECT * FROM perspicuus_events WHERE {where_list} ORDER BY received_at DESC LIMIT ? OFFSET ?',
            list_p + [per_page, offset],
        ).fetchall()
    except sqlite3.OperationalError:
        if status_f in ('ok', 'error'):
            list_c = list(base_c)
            list_p = list(base_p)
            where_list = ' AND '.join(list_c)
            total = db.execute(
                f'SELECT COUNT(*) FROM perspicuus_events WHERE {where_list}', list_p
            ).fetchone()[0]
            rows = db.execute(
                f'SELECT * FROM perspicuus_events WHERE {where_list} ORDER BY received_at DESC LIMIT ? OFFSET ?',
                list_p + [per_page, offset],
            ).fetchall()
        else:
            total, rows = 0, []

    records = []
    for row in rows:
        d = dict(row)
        raw = d.get('inference_json') or '{}'
        try:
            d['inference'] = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            d['inference'] = {}
        d['lat_sum'] = _traits_mean_summary(d['inference'], 'lateral')
        d['post_sum'] = _traits_mean_summary(d['inference'], 'posterior')
        err = d['inference'].get('error')
        if err:
            d['infer_badge'] = 'error'
        elif not d.get('inference_at'):
            d['infer_badge'] = 'pending'
        else:
            d['infer_badge'] = 'ok'
        records.append(d)

    total_pages = max(1, (total + per_page - 1) // per_page) if total else 1

    stations = [r[0] for r in db.execute(
        'SELECT DISTINCT station_id FROM perspicuus_events ORDER BY station_id'
    ).fetchall()]

    infer_stats = {
        'pending': _safe_count(
            f'SELECT COUNT(*) FROM perspicuus_events WHERE {base_where} '
            'AND (inference_at IS NULL OR trim(inference_at) = \'\')',
            base_p,
        ),
        'ok': _safe_count(
            f'SELECT COUNT(*) FROM perspicuus_events WHERE {base_where} '
            'AND inference_at IS NOT NULL AND trim(inference_at) != \'\' '
            'AND json_extract(inference_json, \'$.error\') IS NULL',
            base_p,
        ),
        'error': _safe_count(
            f'SELECT COUNT(*) FROM perspicuus_events WHERE {base_where} '
            "AND json_extract(inference_json, '$.error') IS NOT NULL",
            base_p,
        ),
        'total_all': db.execute('SELECT COUNT(*) FROM perspicuus_events').fetchone()[0],
    }

    return render_template(
        'perspicuus_inferencias.html',
        records=records,
        stations=stations,
        station_filter=station,
        status_filter=status_f,
        q_filter=q,
        page=page,
        total_pages=total_pages,
        total=total,
        infer_stats=infer_stats,
        inference_engine_ready=_perspicuus_inference_engine_ready(),
    )


def _clear_perspicuus_model_slot(role: str) -> None:
    from perspicuus_inference import get_models_dir, load_registry, save_registry

    reg = load_registry()
    old_fn = reg.get(role)
    save_registry({role: None})
    if not old_fn:
        return
    models_dir = os.path.abspath(get_models_dir())
    old_fp = os.path.join(models_dir, os.path.basename(str(old_fn)))
    if old_fp.startswith(models_dir + os.sep) and os.path.isfile(old_fp):
        try:
            os.remove(old_fp)
        except OSError:
            log.warning('Falha ao remover modelo em %s', old_fp)


@app.route('/perspicuus/modelos', methods=['GET', 'POST'])
@login_required
def perspicuus_modelos():
    from perspicuus_inference import (
        get_models_dir,
        load_registry,
        save_registry,
        reset_engine,
        resolve_model_path,
        model_path_source,
        ROLE_TO_ENV,
        get_engine,
    )

    roles = list(ROLE_TO_ENV.keys())
    if request.method == 'POST':
        action = request.form.get('action', 'upload')
        if action == 'clear':
            role = request.form.get('role', '').strip()
            if role not in ROLE_TO_ENV:
                flash('Função inválida.', 'error')
            else:
                _clear_perspicuus_model_slot(role)
                reset_engine()
                flash(
                    'Registo em registry.json atualizado e ficheiro em ml_models removido se existia. '
                    'Se a variável de ambiente estiver definida para este papel, o modelo continua a ser o do ambiente.',
                    'success',
                )
            return redirect(url_for('perspicuus_modelos'))

        role = request.form.get('role', '').strip()
        file = request.files.get('file')
        if role not in ROLE_TO_ENV:
            flash('Função inválida.', 'error')
            return redirect(url_for('perspicuus_modelos'))
        if not file or not file.filename:
            flash('Selecione um ficheiro.', 'error')
            return redirect(url_for('perspicuus_modelos'))
        cl = request.content_length
        if cl is not None and cl > MAX_MODEL_UPLOAD_BYTES:
            flash(
                f'Ficheiro demasiado grande (máx. {MAX_MODEL_UPLOAD_BYTES // (1024 * 1024)} MB).',
                'error',
            )
            return redirect(url_for('perspicuus_modelos'))
        ext = os.path.splitext(file.filename)[1].lower()
        allow = PERSPICUUS_MODEL_ROLE_EXT.get(role, set())
        if ext not in allow:
            flash(
                'Extensão inválida para esta função (permitido: '
                + ', '.join(sorted(allow))
                + ').',
                'error',
            )
            return redirect(url_for('perspicuus_modelos'))
        fname = secure_filename(file.filename)
        if not fname:
            flash('Nome de ficheiro inválido.', 'error')
            return redirect(url_for('perspicuus_modelos'))
        models_dir = get_models_dir()
        dest = os.path.join(models_dir, fname)
        reg = load_registry()
        old_fn = reg.get(role)
        try:
            file.save(dest)
        except OSError as e:
            log.warning('Upload modelo: %s', e)
            flash(f'Erro ao gravar ficheiro: {e}', 'error')
            return redirect(url_for('perspicuus_modelos'))
        save_registry({role: fname})
        if old_fn and os.path.basename(str(old_fn)) != fname:
            old_fp = os.path.join(models_dir, os.path.basename(str(old_fn)))
            abd = os.path.abspath(models_dir) + os.sep
            if old_fp.startswith(abd) and os.path.isfile(old_fp):
                try:
                    os.remove(old_fp)
                except OSError:
                    pass
        reset_engine()
        flash(f'Modelo guardado: {fname}. Sessões ONNX serão recarregadas na próxima inferência.', 'success')
        return redirect(url_for('perspicuus_modelos'))

    reg = load_registry()
    slots = []
    for role in roles:
        p = resolve_model_path(role)
        src = model_path_source(role)
        sz = os.path.getsize(p) if p and os.path.isfile(p) else None
        slots.append({
            'role': role,
            'env_var': ROLE_TO_ENV[role],
            'source': src,
            'path': p,
            'size': sz,
            'registry_name': reg.get(role),
        })
    return render_template(
        'perspicuus_modelos.html',
        slots=slots,
        max_mb=MAX_MODEL_UPLOAD_BYTES // (1024 * 1024),
        ml_models_dir=ML_MODELS_DIR,
        inference_engine_ready=get_engine().is_ready(),
        role_ext=PERSPICUUS_MODEL_ROLE_EXT,
    )


@app.route('/cameras')
@login_required
def cameras():
    # Compatibilidade com links antigos
    return redirect(url_for('calf_monitor'), code=302)

@app.route('/database')
@login_required
def database():
    db      = get_db()
    tab     = request.args.get('tab', 'weather')
    if tab not in ('weather', 'images', 'perspicuus'):
        tab = 'weather'
    page    = max(1, int(request.args.get('page', 1)))
    per_page = 30
    offset  = (page - 1) * per_page
    device_filter = request.args.get('device', '')
    search  = request.args.get('q', '').strip()

    if tab == 'weather':
        conditions, params = ["1=1"], []
        if device_filter:
            conditions.append("device_name = ?"); params.append(device_filter)
        if search:
            conditions.append("(device_name LIKE ? OR raw_json LIKE ?)")
            params += [f'%{search}%', f'%{search}%']
        where = " AND ".join(conditions)

        total = db.execute(f"SELECT COUNT(*) FROM weather WHERE {where}", params).fetchone()[0]
        records = db.execute(
            f"SELECT * FROM weather WHERE {where} ORDER BY received_at DESC LIMIT ? OFFSET ?",
            params + [per_page, offset]
        ).fetchall()
        devices = [r[0] for r in db.execute("SELECT DISTINCT device_name FROM weather ORDER BY device_name").fetchall()]
    elif tab == 'perspicuus':
        conditions, params = ["1=1"], []
        if device_filter:
            conditions.append("station_id = ?"); params.append(device_filter)
        if search:
            like = f'%{search}%'
            conditions.append(
                "(event_id LIKE ? OR station_id LIKE ? OR device_id LIKE ? OR "
                "animal_rfid LIKE ? OR animal_status LIKE ? OR raw_json LIKE ?)"
            )
            params.extend([like, like, like, like, like, like])
        where = " AND ".join(conditions)

        total = db.execute(f"SELECT COUNT(*) FROM perspicuus_events WHERE {where}", params).fetchone()[0]
        records = db.execute(
            f"SELECT * FROM perspicuus_events WHERE {where} ORDER BY timestamp_utc DESC LIMIT ? OFFSET ?",
            params + [per_page, offset]
        ).fetchall()
        devices = [r[0] for r in db.execute(
            "SELECT DISTINCT station_id FROM perspicuus_events ORDER BY station_id"
        ).fetchall()]
    else:
        conditions, params = ["1=1"], []
        if device_filter:
            conditions.append("device_name = ?"); params.append(device_filter)
        if search:
            conditions.append("(device_name LIKE ? OR notes LIKE ?)")
            params += [f'%{search}%', f'%{search}%']
        where = " AND ".join(conditions)

        total = db.execute(f"SELECT COUNT(*) FROM images WHERE {where}", params).fetchone()[0]
        records = db.execute(
            f"SELECT * FROM images WHERE {where} ORDER BY received_at DESC LIMIT ? OFFSET ?",
            params + [per_page, offset]
        ).fetchall()
        devices = [r[0] for r in db.execute("SELECT DISTINCT device_name FROM images ORDER BY device_name").fetchall()]

    total_pages = max(1, (total + per_page - 1) // per_page)
    return render_template('database.html',
        tab=tab, records=records, devices=devices,
        device_filter=device_filter, search=search,
        page=page, total_pages=total_pages, total=total,
    )

# ─── API: INGESTÃO ────────────────────────────────────────────────────────────

@app.route('/api/sensors', methods=['POST'])
@api_auth
def receive_sensors():
    """POST JSON da estação meteorológica ESP32"""
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'JSON inválido'}), 400

    now = datetime.utcnow().isoformat() + 'Z'
    device = data.get('device') or data.get('device_name', 'unknown')

    db = get_db()
    db.execute("""
        INSERT INTO weather
          (received_at, device_name, lux, temp_c, press_hpa, alt_m, humidity, uptime_s, rssi, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now, device,
        data.get('bh1750_lux'),
        _ingest_temp_c(data),
        data.get('bmp280_press_hpa') or data.get('bme280_press_hpa'),
        data.get('bmp280_alt_m') or data.get('bme280_alt_m'),
        _ingest_humidity(data),
        data.get('uptime_s'),
        data.get('rssi'),
        json.dumps(data),
    ))
    db.commit()
    log.info(f"[SENSORS] {device} → lux={data.get('bh1750_lux')} temp={_ingest_temp_c(data)}")
    return jsonify({'status': 'ok', 'received_at': now}), 201

@app.route('/api/upload', methods=['POST'])
@api_auth
def receive_image():
    """POST multipart/form-data do monitor de bezerra ESP32-CAM"""
    if 'image' not in request.files:
        return jsonify({'error': 'Campo "image" ausente'}), 400

    file        = request.files['image']
    device_name = request.form.get('device_name', 'unknown')
    capture_id  = request.form.get('capture_id', 0)
    rssi        = request.form.get('rssi')

    if file.filename == '':
        return jsonify({'error': 'Arquivo vazio'}), 400

    now      = datetime.utcnow()
    ts       = now.strftime('%Y%m%d_%H%M%S')
    safe_dev = secure_filename(device_name)
    filename = f"{safe_dev}_{ts}_{capture_id}.jpg"

    device_dir = os.path.join(UPLOADS_DIR, safe_dev)
    os.makedirs(device_dir, exist_ok=True)
    filepath = os.path.join(device_dir, filename)
    file.save(filepath)
    filesize = os.path.getsize(filepath)

    db = get_db()
    db.execute("""
        INSERT INTO images (received_at, device_name, capture_id, filename, filesize, rssi)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (now.isoformat() + 'Z', device_name, capture_id, filename, filesize, rssi))
    db.commit()
    log.info(f"[UPLOAD] {device_name} → {filename} ({filesize} bytes)")
    return jsonify({'status': 'ok', 'filename': filename, 'size': filesize}), 201


@app.route('/api/perspicuus/events', methods=['POST'])
@api_auth
def receive_perspicuus_event():
    """POST JSON ou multipart (campo form \"json\" + ficheiros frontal_1, lateral_2, …)."""
    payload = None
    files_saved = 0
    ct = (request.content_type or '').lower()
    if 'multipart/form-data' in ct:
        raw = request.form.get('json') or request.form.get('payload') or request.form.get('data')
        if not raw:
            return jsonify({
                'error': 'Multipart: inclua o campo de formulário "json" (string JSON do evento, ex.: images vazio se enviar ficheiros)',
            }), 400
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return jsonify({'error': 'Campo json inválido (não é JSON)'}), 400
        if not isinstance(payload, dict):
            return jsonify({'error': 'Campo json deve ser um objeto'}), 400
        ev = str(payload.get('event_id', '')).strip()
        if not ev:
            return jsonify({'error': 'event_id é obrigatório no JSON ao usar multipart'}), 400
        files_saved = _merge_perspicuus_uploads(payload, request.files, ev)
        if files_saved:
            log.info('[PERSPICUUS] multipart: %s ficheiros gravados em perspicuus/%s', files_saved, secure_filename(ev))
    else:
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            return jsonify({'error': 'JSON inválido'}), 400

    data, err = _normalize_perspicuus_payload(payload)
    if err:
        return jsonify({'error': err}), 400

    now = datetime.utcnow().isoformat() + 'Z'
    db = get_db()
    db.execute("""
        INSERT INTO perspicuus_events (
            event_id, received_at, timestamp_utc, station_id, device_id,
            animal_rfid, animal_status, animal_repetition, inference_ready,
            frontal_json, lateral_json, posterior_json, superior_json,
            total_images, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            received_at=excluded.received_at,
            timestamp_utc=excluded.timestamp_utc,
            station_id=excluded.station_id,
            device_id=excluded.device_id,
            animal_rfid=excluded.animal_rfid,
            animal_status=excluded.animal_status,
            animal_repetition=excluded.animal_repetition,
            inference_ready=excluded.inference_ready,
            frontal_json=excluded.frontal_json,
            lateral_json=excluded.lateral_json,
            posterior_json=excluded.posterior_json,
            superior_json=excluded.superior_json,
            total_images=excluded.total_images,
            raw_json=excluded.raw_json
    """, (
        data['event_id'], now, data['timestamp_utc'], data['station_id'], data['device_id'],
        data['animal_rfid'], data['animal_status'], data['animal_repetition'], data['inference_ready'],
        json.dumps(data['images']['frontal']),
        json.dumps(data['images']['lateral']),
        json.dumps(data['images']['posterior']),
        json.dumps(data['images']['superior']),
        data['total_images'],
        json.dumps(payload),
    ))
    db.commit()
    log.info(
        "[PERSPICUUS] station=%s rfid=%s frames=%s ready=%s",
        data['station_id'], data['animal_rfid'], data['total_images'], bool(data['inference_ready'])
    )
    lat = data['images'].get('lateral') or []
    post = data['images'].get('posterior') or []
    rid_row = db.execute(
        'SELECT id FROM perspicuus_events WHERE event_id = ?',
        (data['event_id'],),
    ).fetchone()
    inference_queued = False
    if rid_row and _perspicuus_auto_infer_enabled() and (lat or post):
        try:
            from perspicuus_inference import get_engine
            if get_engine().is_ready():
                _schedule_perspicuus_auto_infer(rid_row[0])
                inference_queued = True
        except Exception:
            log.debug('auto-infer não agendada (motor indisponível)', exc_info=True)

    out = {
        'status': 'ok',
        'event_id': data['event_id'],
        'station_id': data['station_id'],
        'animal_rfid': data['animal_rfid'],
        'total_images': data['total_images'],
        'inference_ready': bool(data['inference_ready']),
        'received_at': now,
        'files_saved': files_saved,
        'inference_queued': inference_queued,
    }
    return jsonify(out), 201

# ─── API: DADOS PARA FRONTEND ─────────────────────────────────────────────────

@app.route('/api/weather/data')
@login_required
def weather_data():
    db     = get_db()
    device = request.args.get('device', '')
    hours  = min(int(request.args.get('hours', 24)), 720)   # máx 30 dias
    since  = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + 'Z'

    conditions = ["received_at > ?"]
    params = [since]
    if device:
        conditions.append("device_name = ?"); params.append(device)

    where = " AND ".join(conditions)
    rows = db.execute(
        f"SELECT received_at, device_name, lux, temp_c, press_hpa, alt_m, humidity, rssi, raw_json "
        f"FROM weather WHERE {where} ORDER BY received_at ASC LIMIT 1000",
        params
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        t, h = _temp_humidity_for_thi(d)
        if t is not None and h is not None:
            try:
                d['thi'] = round(float(calculate_thi(float(t), float(h))), 2)
            except (TypeError, ValueError):
                d['thi'] = None
        else:
            d['thi'] = None
        d.pop('raw_json', None)
        out.append(d)
    return jsonify(out)

@app.route('/api/calf-monitor/latest')
@app.route('/api/cameras/latest')
@login_required
def calf_monitor_latest():
    db   = get_db()
    rows = db.execute("""
        SELECT i.device_name, i.filename, i.received_at, i.filesize, i.rssi, i.capture_id
        FROM images i
        INNER JOIN (
            SELECT device_name, MAX(received_at) as max_ts FROM images GROUP BY device_name
        ) latest ON i.device_name = latest.device_name AND i.received_at = latest.max_ts
        ORDER BY i.received_at DESC
    """).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/image/<device>/<filename>')
@login_required
def serve_image(device, filename):
    filepath = os.path.join(UPLOADS_DIR, secure_filename(device), secure_filename(filename))
    if not os.path.exists(filepath):
        abort(404)
    return send_file(filepath, mimetype='image/jpeg')


@app.route('/api/perspicuus/media/<event_folder>/<filename>')
@login_required
def serve_perspicuus_media(event_folder, filename):
    """Serve imagens guardadas por POST /api/perspicuus/events (multipart)."""
    base = os.path.abspath(os.path.join(UPLOADS_DIR, 'perspicuus', secure_filename(event_folder)))
    filepath = os.path.abspath(os.path.join(base, secure_filename(filename)))
    if not filepath.startswith(base + os.sep) or not os.path.isfile(filepath):
        abort(404)
    return send_file(filepath)

# ─── API: CRUD ────────────────────────────────────────────────────────────────

@app.route('/api/record/weather/<int:rid>', methods=['DELETE'])
@login_required
def delete_weather(rid):
    db = get_db()
    db.execute("DELETE FROM weather WHERE id = ?", (rid,))
    db.commit()
    return jsonify({'status': 'deleted', 'id': rid})

@app.route('/api/record/weather/<int:rid>', methods=['PATCH'])
@login_required
def edit_weather(rid):
    data    = request.get_json(silent=True) or {}
    allowed = {'lux', 'temp_c', 'press_hpa', 'alt_m', 'humidity', 'rssi', 'device_name'}
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        return jsonify({'error': 'Nenhum campo válido'}), 400
    sets   = ', '.join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [rid]
    db = get_db()
    db.execute(f"UPDATE weather SET {sets} WHERE id=?", values)
    db.commit()
    return jsonify({'status': 'updated'})

@app.route('/api/record/image/<int:rid>', methods=['DELETE'])
@login_required
def delete_image(rid):
    db  = get_db()
    row = db.execute("SELECT filename, device_name FROM images WHERE id=?", (rid,)).fetchone()
    if row:
        fp = os.path.join(UPLOADS_DIR, secure_filename(row['device_name']), secure_filename(row['filename']))
        if os.path.exists(fp):
            os.remove(fp)
        db.execute("DELETE FROM images WHERE id=?", (rid,))
        db.commit()
    return jsonify({'status': 'deleted', 'id': rid})

@app.route('/api/record/image/<int:rid>/notes', methods=['PATCH'])
@login_required
def edit_image_notes(rid):
    data  = request.get_json(silent=True) or {}
    notes = data.get('notes', '')
    db = get_db()
    db.execute("UPDATE images SET notes=? WHERE id=?", (notes, rid))
    db.commit()
    return jsonify({'status': 'updated'})


@app.route('/api/record/perspicuus/<int:rid>', methods=['GET'])
@login_required
def get_perspicuus_record(rid):
    db = get_db()
    row = db.execute("SELECT * FROM perspicuus_events WHERE id = ?", (rid,)).fetchone()
    if not row:
        return jsonify({'error': 'Não encontrado'}), 404
    return jsonify(_serialize_perspicuus_row(row))


def _update_perspicuus_from_normalized(db, rid, data, raw_stored):
    """Atualiza linha por id com payload já normalizado; raw_stored é o JSON gravado em raw_json."""
    ev = data['event_id']
    conflict = db.execute(
        "SELECT id FROM perspicuus_events WHERE event_id = ? AND id != ?",
        (ev, rid),
    ).fetchone()
    if conflict:
        return 'event_id já existe em outro registro'
    db.execute(
        """
        UPDATE perspicuus_events SET
            event_id = ?, timestamp_utc = ?, station_id = ?, device_id = ?,
            animal_rfid = ?, animal_status = ?, animal_repetition = ?, inference_ready = ?,
            frontal_json = ?, lateral_json = ?, posterior_json = ?, superior_json = ?,
            total_images = ?, raw_json = ?
        WHERE id = ?
        """,
        (
            data['event_id'],
            data['timestamp_utc'],
            data['station_id'],
            data['device_id'],
            data['animal_rfid'],
            data['animal_status'],
            data['animal_repetition'],
            int(bool(data['inference_ready'])),
            json.dumps(data['images']['frontal']),
            json.dumps(data['images']['lateral']),
            json.dumps(data['images']['posterior']),
            json.dumps(data['images']['superior']),
            data['total_images'],
            raw_stored,
            rid,
        ),
    )
    return None


@app.route('/api/record/perspicuus/<int:rid>', methods=['PATCH'])
@login_required
def patch_perspicuus_record(rid):
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'JSON inválido'}), 400
    db = get_db()
    row = db.execute("SELECT * FROM perspicuus_events WHERE id = ?", (rid,)).fetchone()
    if not row:
        return jsonify({'error': 'Não encontrado'}), 404

    if 'raw_json' in data or 'payload' in data:
        payload = data.get('raw_json', data.get('payload'))
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return jsonify({'error': 'raw_json não é JSON válido'}), 400
        if not isinstance(payload, dict):
            return jsonify({'error': 'raw_json deve ser um objeto'}), 400
        norm, err = _normalize_perspicuus_payload(payload)
        if err:
            return jsonify({'error': err}), 400
        raw_stored = json.dumps(payload)
        err2 = _update_perspicuus_from_normalized(db, rid, norm, raw_stored)
        if err2:
            return jsonify({'error': err2}), 409
        db.commit()
        updated = db.execute("SELECT * FROM perspicuus_events WHERE id = ?", (rid,)).fetchone()
        return jsonify({'status': 'updated', 'record': _serialize_perspicuus_row(updated)})

    allowed = {
        'station_id': str,
        'device_id': str,
        'animal_rfid': str,
        'animal_status': str,
    }
    int_fields = {'animal_repetition', 'inference_ready'}
    updates = {}
    for key, caster in allowed.items():
        if key not in data:
            continue
        updates[key] = caster(str(data[key]).strip()) if data[key] is not None else ''
    for key in int_fields:
        if key not in data:
            continue
        v = data[key]
        if key == 'inference_ready':
            if isinstance(v, bool):
                updates[key] = 1 if v else 0
            else:
                try:
                    updates[key] = 1 if int(v) else 0
                except (TypeError, ValueError):
                    return jsonify({'error': f'Campo inválido: {key}'}), 400
        else:
            try:
                updates[key] = int(v)
            except (TypeError, ValueError):
                return jsonify({'error': f'Campo inválido: {key}'}), 400

    if not updates:
        return jsonify({'error': 'Nenhum campo válido'}), 400

    sets = ', '.join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [rid]
    db.execute(f"UPDATE perspicuus_events SET {sets} WHERE id = ?", values)
    db.commit()
    updated = db.execute("SELECT * FROM perspicuus_events WHERE id = ?", (rid,)).fetchone()
    return jsonify({'status': 'updated', 'record': _serialize_perspicuus_row(updated)})


@app.route('/api/record/perspicuus/<int:rid>', methods=['DELETE'])
@login_required
def delete_perspicuus_record(rid):
    db = get_db()
    cur = db.execute("DELETE FROM perspicuus_events WHERE id = ?", (rid,))
    db.commit()
    if cur.rowcount == 0:
        return jsonify({'error': 'Não encontrado'}), 404
    return jsonify({'status': 'deleted', 'id': rid})


@app.route('/api/record/perspicuus/<int:rid>/infer', methods=['POST'])
@login_required
def infer_perspicuus_record_api(rid):
    """Executa YOLO + iudicium em frames lateral/posterior com ficheiros no disco."""
    try:
        from perspicuus_inference import run_inference_for_event
    except ImportError as e:
        return jsonify({'error': f'Módulo de inferência indisponível: {e}'}), 503
    db = get_db()
    row = db.execute('SELECT * FROM perspicuus_events WHERE id = ?', (rid,)).fetchone()
    if not row:
        return jsonify({'error': 'Não encontrado'}), 404
    d = dict(row)
    try:
        result = run_inference_for_event(d, UPLOADS_DIR)
        result['trigger'] = 'manual_rerun'
    except Exception as e:
        log.exception('infer perspicuus id=%s', rid)
        return jsonify({'error': str(e)}), 500
    now = datetime.utcnow().isoformat() + 'Z'
    db.execute(
        'UPDATE perspicuus_events SET inference_json = ?, inference_at = ? WHERE id = ?',
        (json.dumps(result, ensure_ascii=False), now, rid),
    )
    db.commit()
    return jsonify({'status': 'ok', 'inference': result}), 200

# ─── DOWNLOADS ────────────────────────────────────────────────────────────────

@app.route('/download/weather')
@login_required
def download_weather():
    db     = get_db()
    device = request.args.get('device', '')
    where  = "WHERE device_name=?" if device else ""
    params = (device,) if device else ()

    rows = db.execute(
        f"SELECT id, received_at, device_name, lux, temp_c, press_hpa, alt_m, humidity, uptime_s, rssi "
        f"FROM weather {where} ORDER BY received_at ASC", params
    ).fetchall()

    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(['id', 'received_at_utc', 'device_name', 'lux_lx',
                'temp_c', 'press_hpa', 'alt_m', 'humidity_pct', 'uptime_s', 'rssi_dbm'])
    for r in rows:
        w.writerow(list(r))

    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'attentus_weather_{ts}.csv'
    )

@app.route('/download/images')
@login_required
def download_images():
    db     = get_db()
    device = request.args.get('device', '')
    where  = "WHERE device_name=?" if device else ""
    params = (device,) if device else ()

    rows = db.execute(
        f"SELECT id, received_at, device_name, capture_id, filename, filesize, rssi, notes "
        f"FROM images {where} ORDER BY received_at ASC", params
    ).fetchall()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        manifest = io.StringIO()
        mw = csv.writer(manifest)
        mw.writerow(['id', 'received_at_utc', 'device_name', 'capture_id',
                     'filename', 'filesize_bytes', 'rssi_dbm', 'notes'])
        for r in rows:
            mw.writerow(list(r))
            fp = os.path.join(UPLOADS_DIR, secure_filename(r['device_name']), secure_filename(r['filename'] or ''))
            if os.path.exists(fp):
                zf.write(fp, f"{r['device_name']}/{r['filename']}")
        zf.writestr('manifest.csv', manifest.getvalue())

    buf.seek(0)
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return send_file(buf, mimetype='application/zip', as_attachment=True,
                     download_name=f'attentus_images_{ts}.zip')


def _perspicuus_download_where():
    """Mesmos filtros que /database?tab=perspicuus (estação + busca)."""
    device = request.args.get('device', '').strip()
    search = request.args.get('q', '').strip()
    conditions, params = ['1=1'], []
    if device:
        conditions.append('station_id = ?')
        params.append(device)
    if search:
        like = f'%{search}%'
        conditions.append(
            '(event_id LIKE ? OR station_id LIKE ? OR device_id LIKE ? OR '
            'animal_rfid LIKE ? OR animal_status LIKE ? OR raw_json LIKE ?)'
        )
        params.extend([like, like, like, like, like, like])
    return ' AND '.join(conditions), params


@app.route('/download/perspicuus')
@login_required
def download_perspicuus():
    where, params = _perspicuus_download_where()
    db = get_db()
    rows = db.execute(
        f"""
        SELECT id, event_id, received_at, timestamp_utc, station_id, device_id,
               animal_rfid, animal_status, animal_repetition, inference_ready,
               total_images
        FROM perspicuus_events
        WHERE {where}
        ORDER BY timestamp_utc ASC
        """,
        params,
    ).fetchall()

    output = io.StringIO()
    w = csv.writer(output)
    w.writerow([
        'id', 'event_id', 'received_at_utc', 'timestamp_utc', 'station_id', 'device_id',
        'animal_rfid', 'animal_status', 'animal_repetition', 'inference_ready', 'total_images',
    ])
    for r in rows:
        w.writerow(list(r))

    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'attentus_perspicuus_{ts}.csv',
    )


@app.route('/download/perspicuus/event/<int:rid>')
@login_required
def download_perspicuus_event(rid):
    db = get_db()
    row = db.execute('SELECT * FROM perspicuus_events WHERE id = ?', (rid,)).fetchone()
    if not row:
        abort(404)
    payload = _serialize_perspicuus_row(row)
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    buf = io.BytesIO(raw.encode('utf-8'))
    buf.seek(0)
    ev = dict(row).get('event_id') or str(rid)
    safe = secure_filename(str(ev).replace(' ', '_'))[:80] or f'event_{rid}'
    return send_file(
        buf,
        mimetype='application/json',
        as_attachment=True,
        download_name=f'perspicuus_{safe}.json',
    )

# ─── HEALTHCHECK ──────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    now_utc = datetime.now(timezone.utc)
    return jsonify({
        'status': 'ok',
        'ts': now_utc.isoformat().replace('+00:00', 'Z'),
        'ts_br': format_br_datetime(now_utc.isoformat().replace('+00:00', 'Z'), '%Y-%m-%d %H:%M:%S'),
    }), 200

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
