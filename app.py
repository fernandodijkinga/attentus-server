"""
============================================================
  ATTENTUS — Servidor Central de Coleta de Dados
  GenMate Field Intelligence Platform
============================================================
  Recebe dados de estações meteorológicas (ESP32+DHT22+BMP280+BH1750)
  e câmeras de baia (ESP32-CAM) via HTTP.

  Endpoints de ingestão:
    POST /api/sensors   → JSON da estação meteorológica
    POST /api/upload    → multipart/form-data da câmera

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
import sqlite3
import json
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
log.info(f"DATA_DIR={DATA_DIR}")

# Autenticação de usuário web
ADMIN_USER      = os.environ.get('ADMIN_USER', 'admin')
ADMIN_PASS_HASH = generate_password_hash(os.environ.get('ADMIN_PASS', 'attentus2024'))

# Chave de API para dispositivos (vazio = sem restrição)
API_KEY = os.environ.get('API_KEY', '')


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


ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
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

    CREATE INDEX IF NOT EXISTS idx_weather_received ON weather(received_at);
    CREATE INDEX IF NOT EXISTS idx_weather_device   ON weather(device_name);
    CREATE INDEX IF NOT EXISTS idx_images_received  ON images(received_at);
    CREATE INDEX IF NOT EXISTS idx_images_device    ON images(device_name);
    """)
    db.commit()
    db.close()
    log.info("DB inicializado OK")

init_db()

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

@app.route('/cameras')
@login_required
def cameras():
    db = get_db()
    # Última imagem por câmera
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

    # Contagem total por câmera
    counts = dict(db.execute(
        "SELECT device_name, COUNT(*) FROM images GROUP BY device_name"
    ).fetchall())

    return render_template('cameras.html', cameras=rows, counts=counts)

@app.route('/database')
@login_required
def database():
    db      = get_db()
    tab     = request.args.get('tab', 'weather')
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
    """POST multipart/form-data da câmera ESP32-CAM"""
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

@app.route('/api/cameras/latest')
@login_required
def cameras_latest():
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
