"""
Rotas GenMate Black Barn (Holstein: imagem/vídeo, segmentação, pose, correlações).
Registo: register_black_barn(app) chamado a partir de app.py.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from flask import abort, flash, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

log = logging.getLogger(__name__)


def register_black_barn(app) -> None:
    import app as main

    def _bb_upload_root() -> str:
        return main.BLACK_BARN_UPLOADS_DIR

    def _schedule_job(record_id: int) -> None:
        threading.Thread(
            target=_run_bb_worker,
            args=(record_id, main.DB_PATH, main.UPLOADS_DIR),
            daemon=True,
        ).start()

    @app.route("/genmate-black-barn/importar", methods=["GET", "POST"])
    @main.login_required
    def black_barn_importar():
        db = main.get_db()
        if request.method == "POST":
            farm_id = secure_filename((request.form.get("farm_id") or "default").strip()) or "default"
            lot_id = (request.form.get("lot_id") or "").strip()[:128]
            animal_tag = (request.form.get("animal_tag") or "").strip()[:128]
            kind = (request.form.get("kind") or "image").strip()
            view_mode = (request.form.get("view_mode") or "auto").strip().lower()
            if kind not in ("image", "video_single", "video_dual"):
                flash("Tipo de importação inválido.", "error")
                return redirect(url_for("black_barn_importar"))
            if not animal_tag:
                flash("Brinco é obrigatório.", "error")
                return redirect(url_for("black_barn_importar"))
            inferred_view = ""
            if view_mode in ("lateral", "posterior"):
                inferred_view = view_mode
            os.makedirs(os.path.join(_bb_upload_root(), farm_id), exist_ok=True)

            def save_upload(field: str) -> tuple[str, str]:
                f = request.files.get(field)
                if not f or not f.filename:
                    return "", ""
                ext = os.path.splitext(f.filename)[1].lower()
                if ext not in main.BLACK_BARN_MEDIA_EXTS:
                    raise ValueError(f"extensão não permitida: {ext}")
                fn = f"{uuid.uuid4().hex}{ext}"
                disk = os.path.join(_bb_upload_root(), farm_id, fn)
                f.save(disk)
                pub = f"/api/black-barn/media/{farm_id}/{fn}"
                return disk, pub

            try:
                path_lat = path_post = path_single = ""
                pub_lat = pub_post = pub_single = ""
                if kind == "image":
                    path_single, pub_single = save_upload("media_file")
                    if not path_single:
                        raise ValueError("selecione uma imagem")
                elif kind == "video_single":
                    path_single, pub_single = save_upload("media_file")
                    if not path_single:
                        raise ValueError("selecione um vídeo")
                else:
                    path_lat, pub_lat = save_upload("video_lateral")
                    path_post, pub_post = save_upload("video_posterior")
                    if not path_lat or not path_post:
                        raise ValueError("dois vídeos obrigatórios (lateral e posterior)")
                now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                cur = db.execute(
                    """
                    INSERT INTO black_barn_records (
                        created_at, farm_id, lot_id, animal_tag, kind,
                        path_lateral, path_posterior, path_single,
                        public_lateral, public_posterior, public_single,
                        inferred_view, status, result_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'processing', '{}')
                    """,
                    (
                        now, farm_id, lot_id, animal_tag, kind,
                        path_lat, path_post, path_single,
                        pub_lat, pub_post, pub_single,
                        inferred_view,
                    ),
                )
                db.commit()
                rid = int(cur.lastrowid)
                _schedule_job(rid)
                flash(f"Registo #{rid} em processamento.", "success")
                return redirect(url_for("black_barn_importar"))
            except ValueError as e:
                flash(str(e), "error")
                return redirect(url_for("black_barn_importar"))

        rows = db.execute(
            """
            SELECT id, created_at, farm_id, lot_id, animal_tag, kind, status, inferred_view, error_text
            FROM black_barn_records
            ORDER BY id DESC
            LIMIT 40
            """
        ).fetchall()
        return render_template("black_barn_importar.html", records=[dict(x) for x in rows])

    @app.route("/genmate-black-barn/dataset")
    @main.login_required
    def black_barn_dataset():
        db = main.get_db()
        farm = request.args.get("farm", "").strip()
        lot = request.args.get("lot", "").strip()
        q = request.args.get("q", "").strip()
        cond, params = ["1=1"], []
        if farm:
            cond.append("farm_id = ?")
            params.append(farm)
        if lot:
            cond.append("lot_id LIKE ?")
            params.append(f"%{lot}%")
        if q:
            cond.append("(animal_tag LIKE ? OR CAST(id AS TEXT) LIKE ?)")
            params.extend([f"%{q}%", f"%{q}%"])
        where = " AND ".join(cond)
        rows = db.execute(
            f"""
            SELECT * FROM black_barn_records
            WHERE {where}
            ORDER BY id DESC
            LIMIT 800
            """,
            params,
        ).fetchall()
        farms = [r[0] for r in db.execute("SELECT DISTINCT farm_id FROM black_barn_records ORDER BY farm_id LIMIT 200")]
        return render_template(
            "black_barn_dataset.html",
            records=[dict(x) for x in rows],
            farms=farms,
            farm_filter=farm,
            lot_filter=lot,
            q_filter=q,
        )

    @app.route("/genmate-black-barn/modelos", methods=["GET", "POST"])
    @main.admin_required
    def black_barn_modelos():
        from perspicuus_inference import (
            get_models_dir,
            load_registry,
            model_path_source,
            reset_engine,
            resolve_model_path,
            ROLE_TO_ENV,
            save_registry,
            get_engine,
        )

        roles = [
            "bb_yolo", "bb_identification", "bb_seg", "bb_pose",
            "bb_lateral", "bb_posterior", "bb_lateral_meta", "bb_posterior_meta",
        ]
        role_labels = {
            "bb_yolo": "YOLO deteção / crop (ONNX) — Perspicuus",
            "bb_identification": "YOLO identificação de vista (.pt Ultralytics ou ONNX)",
            "bb_seg": "YOLO segmentação (.pt Ultralytics)",
            "bb_pose": "YOLO pose — classes cow / UC (.pt Ultralytics)",
            "bb_lateral": "Perspicuus lateral (ONNX)",
            "bb_posterior": "Perspicuus posterior (ONNX)",
            "bb_lateral_meta": "Metadata JSON — lateral",
            "bb_posterior_meta": "Metadata JSON — posterior",
        }
        if request.method == "POST":
            action = request.form.get("action", "upload")
            role = request.form.get("role", "").strip()
            if role not in roles:
                flash("Função inválida.", "error")
                return redirect(url_for("black_barn_modelos"))
            if action == "clear":
                main._clear_perspicuus_model_slot(role)
                reset_engine("black_barn")
                flash("Slot limpo.", "success")
                return redirect(url_for("black_barn_modelos"))
            file = request.files.get("file")
            if not file or not file.filename:
                flash("Selecione um ficheiro.", "error")
                return redirect(url_for("black_barn_modelos"))
            ext = os.path.splitext(file.filename)[1].lower()
            allow = main.PERSPICUUS_MODEL_ROLE_EXT.get(role, set())
            if ext not in allow:
                flash("Extensão inválida.", "error")
                return redirect(url_for("black_barn_modelos"))
            fname = secure_filename(file.filename)
            if not fname:
                flash("Nome inválido.", "error")
                return redirect(url_for("black_barn_modelos"))
            dest = os.path.join(get_models_dir(), fname)
            file.save(dest)
            save_registry({role: fname})
            reset_engine("black_barn")
            flash(f"Modelo guardado: {fname}", "success")
            return redirect(url_for("black_barn_modelos"))

        reg = load_registry()
        slots = []
        for role in roles:
            p = resolve_model_path(role)
            src = model_path_source(role)
            sz = os.path.getsize(p) if p and os.path.isfile(p) else None
            slots.append({
                "role": role,
                "env_var": ROLE_TO_ENV[role],
                "source": src,
                "path": p,
                "size": sz,
                "registry_name": reg.get(role),
            })
        eng = get_engine("black_barn")
        ready = eng.is_ready() and bool(eng.onnx_path_for("lateral")) and bool(eng.onnx_path_for("posterior"))
        return render_template(
            "black_barn_modelos.html",
            slots=slots,
            max_mb=main.MAX_MODEL_UPLOAD_BYTES // (1024 * 1024),
            ml_models_dir=main.ML_MODELS_DIR,
            inference_engine_ready=ready,
            role_ext=main.PERSPICUUS_MODEL_ROLE_EXT,
            role_labels=role_labels,
        )

    @app.route("/genmate-black-barn/segmentacao")
    @main.login_required
    def black_barn_segmentacao():
        db = main.get_db()
        rows = db.execute(
            "SELECT id, farm_id, lot_id, animal_tag, kind, status, public_single, public_lateral, public_posterior FROM black_barn_records ORDER BY id DESC LIMIT 200"
        ).fetchall()
        defs = db.execute(
            "SELECT id, trait_key, label, source, config_json FROM black_barn_trait_defs WHERE source = 'seg' ORDER BY id DESC LIMIT 80"
        ).fetchall()
        return render_template(
            "black_barn_segmentacao.html",
            records=[dict(x) for x in rows],
            trait_defs=[dict(x) for x in defs],
        )

    @app.route("/genmate-black-barn/keypoints")
    @main.login_required
    def black_barn_keypoints():
        db = main.get_db()
        rows = db.execute(
            "SELECT id, farm_id, lot_id, animal_tag, kind, status, public_single, public_lateral, public_posterior FROM black_barn_records ORDER BY id DESC LIMIT 200"
        ).fetchall()
        defs = db.execute(
            "SELECT id, trait_key, label, source, config_json FROM black_barn_trait_defs WHERE source = 'kp' ORDER BY id DESC LIMIT 80"
        ).fetchall()
        return render_template(
            "black_barn_keypoints.html",
            records=[dict(x) for x in rows],
            trait_defs=[dict(x) for x in defs],
        )

    @app.route("/genmate-black-barn/individual/<int:rid>")
    @main.login_required
    def black_barn_individual(rid: int):
        db = main.get_db()
        row = db.execute("SELECT * FROM black_barn_records WHERE id = ?", (rid,)).fetchone()
        if not row:
            abort(404)
        r = dict(row)
        try:
            r["result_parsed"] = json.loads(r.get("result_json") or "{}")
        except json.JSONDecodeError:
            r["result_parsed"] = {}
        traits = db.execute(
            "SELECT trait_key, value, frame_index FROM black_barn_trait_values WHERE record_id = ? ORDER BY trait_key",
            (rid,),
        ).fetchall()
        return render_template("black_barn_individual.html", record=r, trait_rows=[dict(x) for x in traits])

    @app.route("/genmate-black-barn/correlacoes")
    @main.login_required
    def black_barn_correlacoes():
        db = main.get_db()
        keys = _collect_trait_keys(db)
        return render_template("black_barn_correlacoes.html", trait_keys=keys)

    @app.route("/api/black-barn/media/<path:farm>/<path:filename>")
    @main.login_required
    def serve_black_barn_media(farm, filename):
        farm_s = secure_filename(farm)
        fn = secure_filename(filename)
        if not farm_s or not fn:
            abort(404)
        root = os.path.abspath(_bb_upload_root())
        d = os.path.abspath(os.path.join(root, farm_s))
        if not d.startswith(root + os.sep):
            abort(404)
        return send_from_directory(d, fn, as_attachment=False)

    @app.route("/api/genmate-black-barn/correlations/points")
    @main.login_required
    def api_black_barn_correlation_points():
        xk = request.args.get("x", "").strip()
        yk = request.args.get("y", "").strip()
        if not xk or not yk or xk == yk:
            return jsonify({"error": "parâmetros_x_y"}), 400
        db = main.get_db()
        pts, nx, ny = _paired_trait_points(db, xk, yk)
        if len(pts) < 2:
            return jsonify({"n": len(pts), "points": pts, "pearson": None, "spearman": None, "kendall": None})
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pear = _pearson(xs, ys)
        spr, ken = _rank_correlations(xs, ys)
        return jsonify({
            "n": len(pts),
            "points": pts,
            "pearson": pear,
            "spearman": spr,
            "kendall": ken,
            "label_x": nx,
            "label_y": ny,
        })

    @app.route("/api/genmate-black-barn/trait-defs", methods=["GET", "POST"])
    @main.login_required
    def api_black_barn_trait_defs():
        db = main.get_db()
        if request.method == "GET":
            rows = db.execute(
                "SELECT id, farm_id, trait_key, label, source, config_json FROM black_barn_trait_defs ORDER BY source, trait_key LIMIT 500"
            ).fetchall()
            return jsonify({"defs": [dict(x) for x in rows]})
        data = request.get_json(silent=True) or {}
        farm_id = secure_filename(str(data.get("farm_id") or "").strip() or "default")
        trait_key = str(data.get("trait_key") or "").strip()[:80]
        label = str(data.get("label") or trait_key)[:160]
        source = str(data.get("source") or "seg").strip()[:16]
        if source not in ("seg", "kp", "persp"):
            return jsonify({"error": "source"}), 400
        if not trait_key:
            return jsonify({"error": "trait_key"}), 400
        cfg = json.dumps(data.get("config") or {}, ensure_ascii=False)
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        try:
            db.execute(
                """
                INSERT INTO black_barn_trait_defs (created_at, farm_id, trait_key, label, source, config_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (now, farm_id, trait_key, label, source, cfg),
            )
            db.commit()
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"ok": True})

    @app.route("/api/genmate-black-barn/trait-value", methods=["POST"])
    @main.login_required
    def api_black_barn_trait_value():
        db = main.get_db()
        data = request.get_json(silent=True) or {}
        rid = int(data.get("record_id") or 0)
        trait_key = str(data.get("trait_key") or "").strip()[:80]
        val = data.get("value")
        if rid <= 0 or not trait_key or val is None:
            return jsonify({"error": "dados"}), 400
        try:
            v = float(val)
        except (TypeError, ValueError):
            return jsonify({"error": "value"}), 400
        frame_index = data.get("frame_index")
        fi = int(frame_index) if frame_index is not None and str(frame_index).isdigit() else None
        db.execute(
            "DELETE FROM black_barn_trait_values WHERE record_id = ? AND trait_key = ? AND IFNULL(frame_index, -999) = IFNULL(?, -999)",
            (rid, trait_key, fi),
        )
        db.execute(
            "INSERT INTO black_barn_trait_values (record_id, trait_key, frame_index, value) VALUES (?, ?, ?, ?)",
            (rid, trait_key, fi, v),
        )
        db.commit()
        return jsonify({"ok": True})


def _run_bb_worker(record_id: int, db_path: str, uploads_dir: str) -> None:
    from black_barn_inference import process_record_on_disk

    process_record_on_disk(record_id, db_path, uploads_dir)
    rows = db.execute(
        "SELECT id, result_json FROM black_barn_records WHERE status = 'done' ORDER BY id DESC LIMIT 500"
    ).fetchall()
    keys: dict[str, str] = {}
    for row in rows:
        try:
            rj = json.loads(row["result_json"] or "{}")
        except json.JSONDecodeError:
            continue
        p = rj.get("perspicuus")
        if isinstance(p, dict):
            traits = p.get("traits")
            if isinstance(traits, dict):
                for k in traits:
                    keys[f"persp:{k}"] = f"Perspicuus · {k}"
            frames = p.get("frames")
            if isinstance(frames, list):
                for fr in frames:
                    if isinstance(fr, dict) and isinstance(fr.get("traits"), dict):
                        for k in fr["traits"]:
                            keys[f"persp:{k}"] = f"Perspicuus · {k}"
    tvs = db.execute("SELECT DISTINCT trait_key FROM black_barn_trait_values LIMIT 500").fetchall()
    for (k,) in tvs:
        if k:
            keys[f"custom:{k}"] = f"Medida custom · {k}"
    defs = db.execute("SELECT trait_key, label, source FROM black_barn_trait_defs LIMIT 300").fetchall()
    for d in defs:
        tid = f"custom:{d['trait_key']}"
        keys.setdefault(tid, d["label"] or tid)
    return [{"id": k, "label": v} for k, v in sorted(keys.items(), key=lambda x: x[0].lower())]


def _paired_trait_points(db, xk: str, yk: str) -> tuple[list[list[float]], str, str]:
    """Extrai pares (x,y) por registo onde ambas as medidas existem."""
    rows = db.execute(
        "SELECT id, result_json FROM black_barn_records WHERE status = 'done' ORDER BY id DESC LIMIT 500"
    ).fetchall()
    pts: list[list[float]] = []

    def get_val(parsed: dict, key: str) -> float | None:
        if key.startswith("persp:"):
            tname = key.split(":", 1)[1]
            p = parsed.get("perspicuus") or {}
            if not isinstance(p, dict):
                return None
            tr = p.get("traits")
            if isinstance(tr, dict) and tname in tr:
                try:
                    return float(tr[tname])
                except (TypeError, ValueError):
                    pass
            frames = p.get("frames")
            if isinstance(frames, list):
                for fr in frames:
                    if not isinstance(fr, dict):
                        continue
                    tr2 = fr.get("traits")
                    if isinstance(tr2, dict) and tname in tr2:
                        try:
                            return float(tr2[tname])
                        except (TypeError, ValueError):
                            continue
            return None
        return None

    for row in rows:
        rid = row["id"]
        try:
            parsed = json.loads(row["result_json"] or "{}")
        except json.JSONDecodeError:
            continue
        parsed = {**parsed, "_record_id": rid}
        xv = get_val(parsed, xk)
        yv = get_val(parsed, yk)
        if xv is None:
            xv = _trait_value_from_db(db, rid, xk)
        if yv is None:
            yv = _trait_value_from_db(db, rid, yk)
        if xv is not None and yv is not None and not (math.isnan(xv) or math.isnan(yv)):
            pts.append([round(xv, 6), round(yv, 6)])
    return pts, xk, yk


def _trait_value_from_db(db, record_id: int, key: str) -> float | None:
    if not key.startswith("custom:"):
        return None
    tname = key.split(":", 1)[1]
    row = db.execute(
        "SELECT value FROM black_barn_trait_values WHERE record_id = ? AND trait_key = ? AND frame_index IS NULL LIMIT 1",
        (record_id, tname),
    ).fetchone()
    if not row:
        return None
    try:
        return float(row["value"])
    except (TypeError, ValueError):
        return None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    r = num / (denx * deny)
    return round(r, 6)


def _rank_correlations(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    try:
        from scipy import stats  # type: ignore

        sp = stats.spearmanr(xs, ys).correlation
        kd = stats.kendalltau(xs, ys).correlation
        return (round(float(sp), 6) if sp == sp else None, round(float(kd), 6) if kd == kd else None)
    except Exception:
        return None, None
