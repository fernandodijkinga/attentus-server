"""
Rotas GenMate Black Barn (Holstein: imagem/vídeo, segmentação, pose, correlações).
Registo: register_black_barn(app) chamado a partir de app.py.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from flask import abort, flash, jsonify, redirect, render_template, request, send_from_directory, url_for, Response
from werkzeug.utils import secure_filename

log = logging.getLogger(__name__)

BB_REPROCESS_JOBS: Dict[str, Dict[str, Any]] = {}
BB_REPROCESS_LOCK = threading.Lock()


def _bb_init_reprocess_job(*, farm_id: str, total: int, started_by: str) -> str:
    job_id = uuid.uuid4().hex
    with BB_REPROCESS_LOCK:
        BB_REPROCESS_JOBS[job_id] = {
            "job_id": job_id,
            "farm_id_filter": farm_id or None,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "finished_at": None,
            "started_by": started_by,
            "total": int(total),
            "processed": 0,
            "ok": 0,
            "errors": 0,
            "current_record_id": None,
            "current_step": "A preparar reprocessamento Black Barn",
            "sample_errors": [],
        }
    return job_id


def _bb_update_reprocess_job(job_id: str, **fields: Any) -> None:
    with BB_REPROCESS_LOCK:
        job = BB_REPROCESS_JOBS.get(job_id)
        if not job:
            return
        job.update(fields)


def _bb_run_reprocess_job(job_id: str, ids: List[int], db_path: str, uploads_dir: str) -> None:
    from black_barn_inference import process_record_on_disk
    from perspicuus_inference import reset_engine

    reset_engine("black_barn")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ok_n = 0
    err_n = 0
    total = len(ids)
    try:
        for i, rid in enumerate(ids, start=1):
            _bb_update_reprocess_job(
                job_id,
                processed=i - 1,
                current_record_id=int(rid),
                current_step=f"YOLO + segmentação + pose + Perspicuus · {i}/{total}",
            )
            try:
                conn.execute(
                    "UPDATE black_barn_records SET status = ?, error_text = NULL WHERE id = ?",
                    ("processing", int(rid)),
                )
                conn.commit()
            except sqlite3.Error:
                log.warning("[BlackBarn] não foi possível marcar processing id=%s", rid)
            try:
                process_record_on_disk(int(rid), db_path, uploads_dir)
            except Exception as ex:  # noqa: BLE001
                log.exception("[BlackBarn] reprocess id=%s", rid)
                err_n += 1
                try:
                    conn.execute(
                        "UPDATE black_barn_records SET status = ?, error_text = ? WHERE id = ?",
                        ("error", str(ex), int(rid)),
                    )
                    conn.commit()
                except sqlite3.Error:
                    pass
                with BB_REPROCESS_LOCK:
                    job = BB_REPROCESS_JOBS.get(job_id)
                    if job and len(job.get("sample_errors") or []) < 25:
                        se = list(job.get("sample_errors") or [])
                        se.append({"id": int(rid), "error": str(ex)})
                        job["sample_errors"] = se
                _bb_update_reprocess_job(job_id, processed=i, ok=ok_n, errors=err_n)
                continue
            row = conn.execute(
                "SELECT status, error_text FROM black_barn_records WHERE id = ?",
                (int(rid),),
            ).fetchone()
            if row and str(row["status"] or "") == "done":
                ok_n += 1
            else:
                err_n += 1
                with BB_REPROCESS_LOCK:
                    job = BB_REPROCESS_JOBS.get(job_id)
                    if job and len(job.get("sample_errors") or []) < 25:
                        se = list(job.get("sample_errors") or [])
                        se.append({
                            "id": int(rid),
                            "error": str(row["error_text"] or row["status"] or "erro"),
                        })
                        job["sample_errors"] = se
            _bb_update_reprocess_job(job_id, processed=i, ok=ok_n, errors=err_n)
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        _bb_update_reprocess_job(
            job_id,
            status="done",
            finished_at=now,
            current_step="Concluído",
            current_record_id=None,
            processed=total,
            ok=ok_n,
            errors=err_n,
        )
    except Exception as e:
        log.exception("[BlackBarn] job reprocess falhou job=%s", job_id)
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with BB_REPROCESS_LOCK:
            j = BB_REPROCESS_JOBS.get(job_id) or {}
            proc = int(j.get("processed") or 0)
        _bb_update_reprocess_job(
            job_id,
            status="failed",
            finished_at=now,
            current_step=f"Falhou: {e}",
            current_record_id=None,
            processed=min(total, proc),
            ok=ok_n,
            errors=err_n,
        )
    finally:
        conn.close()


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
            "bb_yolo": "YOLO vista + crop (ONNX) — igual Perspicuus: bbox e classe lateral/posterior",
            "bb_identification": "Opcional: .pt ou ONNX só para votar vista (se vazio, usa o ficheiro de bb_yolo)",
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
        from black_barn_inference import bb_preview_frame_count_for_record

        db = main.get_db()
        rows = db.execute(
            "SELECT id, farm_id, lot_id, animal_tag, kind, status, public_single, public_lateral, public_posterior FROM black_barn_records ORDER BY id DESC LIMIT 200"
        ).fetchall()
        defs = db.execute(
            "SELECT id, trait_key, label, source, config_json FROM black_barn_trait_defs WHERE source = 'seg' ORDER BY id DESC LIMIT 80"
        ).fetchall()
        recs: list[dict[str, Any]] = []
        for x in rows:
            d = dict(x)
            d["preview_frame_count"] = bb_preview_frame_count_for_record(d, main.UPLOADS_DIR, "auto")
            recs.append(d)
        _seg_u = url_for("api_black_barn_preview_segmentation", rid=0)
        seg_preview_base = _seg_u.rsplit("/", 1)[0] + "/"
        return render_template(
            "black_barn_segmentacao.html",
            records=recs,
            trait_defs=[dict(x) for x in defs],
            seg_preview_base=seg_preview_base,
        )

    @app.route("/genmate-black-barn/keypoints")
    @main.login_required
    def black_barn_keypoints():
        from black_barn_inference import bb_preview_frame_count_for_record

        db = main.get_db()
        rows = db.execute(
            "SELECT id, farm_id, lot_id, animal_tag, kind, status, public_single, public_lateral, public_posterior FROM black_barn_records ORDER BY id DESC LIMIT 200"
        ).fetchall()
        defs = db.execute(
            "SELECT id, trait_key, label, source, config_json FROM black_barn_trait_defs WHERE source = 'kp' ORDER BY id DESC LIMIT 80"
        ).fetchall()
        recs = []
        for x in rows:
            d = dict(x)
            d["preview_frame_count"] = bb_preview_frame_count_for_record(d, main.UPLOADS_DIR, "auto")
            recs.append(d)
        _pose_u = url_for("api_black_barn_preview_pose", rid=0)
        pose_preview_base = _pose_u.rsplit("/", 1)[0] + "/"
        return render_template(
            "black_barn_keypoints.html",
            records=recs,
            trait_defs=[dict(x) for x in defs],
            pose_preview_base=pose_preview_base,
        )

    @app.route("/genmate-black-barn/individual/<int:rid>")
    @main.login_required
    def black_barn_individual(rid: int):
        db = main.get_db()
        row = db.execute("SELECT * FROM black_barn_records WHERE id = ?", (rid,)).fetchone()
        if not row:
            abort(404)
        r = dict(row)
        r["result_parsed"] = _bb_parse_result_json(r.get("result_json"))
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

    @app.route("/api/genmate-black-barn/preview/segmentation/<int:rid>", methods=["GET"])
    @main.login_required
    def api_black_barn_preview_segmentation(rid: int):
        from perspicuus_inference import resolve_model_path

        from black_barn_inference import (
            bb_load_frame_bgr_for_record,
            bb_png_message_bytes,
            bb_render_segmentation_plot_png,
        )

        frame = int(request.args.get("frame", 0) or 0)
        clip = str(request.args.get("clip", "auto") or "auto").strip().lower()
        if clip not in ("auto", "single", "lateral", "posterior"):
            clip = "auto"
        db = main.get_db()
        row = db.execute("SELECT * FROM black_barn_records WHERE id = ?", (rid,)).fetchone()
        if not row:
            return Response(bb_png_message_bytes("registo não encontrado"), mimetype="image/png")
        rec = dict(row)
        img = bb_load_frame_bgr_for_record(rec, main.UPLOADS_DIR, frame, clip)
        if img is None:
            return Response(bb_png_message_bytes("sem mídia ou frame inválido"), mimetype="image/png")
        seg = resolve_model_path("bb_seg")
        if not seg or not os.path.isfile(seg):
            return Response(bb_png_message_bytes("modelo bb_seg (.pt) não configurado"), mimetype="image/png")
        try:
            png = bb_render_segmentation_plot_png(seg, img)
        except Exception as ex:  # noqa: BLE001
            log.exception("[BlackBarn] preview segmentação id=%s", rid)
            png = bb_png_message_bytes(str(ex)[:200])
        return Response(png, mimetype="image/png")

    @app.route("/api/genmate-black-barn/preview/pose/<int:rid>", methods=["GET"])
    @main.login_required
    def api_black_barn_preview_pose(rid: int):
        from perspicuus_inference import resolve_model_path

        from black_barn_inference import (
            bb_load_frame_bgr_for_record,
            bb_png_message_bytes,
            bb_render_pose_plot_png,
        )

        frame = int(request.args.get("frame", 0) or 0)
        clip = str(request.args.get("clip", "auto") or "auto").strip().lower()
        if clip not in ("auto", "single", "lateral", "posterior"):
            clip = "auto"
        db = main.get_db()
        row = db.execute("SELECT * FROM black_barn_records WHERE id = ?", (rid,)).fetchone()
        if not row:
            return Response(bb_png_message_bytes("registo não encontrado"), mimetype="image/png")
        rec = dict(row)
        img = bb_load_frame_bgr_for_record(rec, main.UPLOADS_DIR, frame, clip)
        if img is None:
            return Response(bb_png_message_bytes("sem mídia ou frame inválido"), mimetype="image/png")
        pose = resolve_model_path("bb_pose")
        if not pose or not os.path.isfile(pose):
            return Response(bb_png_message_bytes("modelo bb_pose (.pt) não configurado"), mimetype="image/png")
        try:
            png = bb_render_pose_plot_png(pose, img)
        except Exception as ex:  # noqa: BLE001
            log.exception("[BlackBarn] preview pose id=%s", rid)
            png = bb_png_message_bytes(str(ex)[:200])
        return Response(png, mimetype="image/png")

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

    @app.route("/api/genmate-black-barn/reprocess-all/start", methods=["POST"])
    @main.login_required
    def api_black_barn_reprocess_all_start():
        data = request.get_json(silent=True) or {}
        farm = str(data.get("farm_id") or request.args.get("farm_id", "") or "").strip()
        max_n = max(1, min(10000, int(os.environ.get("BB_REPROCESS_MAX", "5000"))))
        conn = sqlite3.connect(main.DB_PATH)
        conn.row_factory = sqlite3.Row
        cond, params = ["1=1"], []
        if farm:
            cond.append("farm_id = ?")
            params.append(farm)
        where = " AND ".join(cond)
        ids = [
            int(r["id"])
            for r in conn.execute(
                f"SELECT id FROM black_barn_records WHERE {where} ORDER BY id ASC LIMIT ?",
                params + [max_n],
            ).fetchall()
        ]
        conn.close()
        job_id = _bb_init_reprocess_job(
            farm_id=farm,
            total=len(ids),
            started_by=str(main.session.get("username") or "unknown"),
        )
        if not ids:
            _bb_update_reprocess_job(
                job_id,
                status="done",
                finished_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                current_step="Sem registos para reprocessar",
                processed=0,
                total=0,
            )
            return jsonify({"status": "done", "job_id": job_id, "total": 0})
        threading.Thread(
            target=_bb_run_reprocess_job,
            args=(job_id, ids, main.DB_PATH, main.UPLOADS_DIR),
            daemon=True,
        ).start()
        return jsonify({"status": "started", "job_id": job_id, "total": len(ids), "farm_id_filter": farm or None})

    @app.route("/api/genmate-black-barn/reprocess-all/status/<job_id>", methods=["GET"])
    @main.login_required
    def api_black_barn_reprocess_all_status(job_id: str):
        with BB_REPROCESS_LOCK:
            job = dict(BB_REPROCESS_JOBS.get(job_id) or {})
        if not job:
            return jsonify({"error": "job_not_found"}), 404
        total = int(job.get("total") or 0)
        processed = int(job.get("processed") or 0)
        pct = int(round((processed / total) * 100)) if total > 0 else 100
        job["progress_pct"] = max(0, min(100, pct))
        return jsonify(job)


def _run_bb_worker(record_id: int, db_path: str, uploads_dir: str) -> None:
    from black_barn_inference import process_record_on_disk

    process_record_on_disk(record_id, db_path, uploads_dir)


def _bb_parse_result_json(raw: Any) -> dict[str, Any]:
    """Garante dict para `result_json`; evita 500 se o JSON for lista/número ou inválido."""
    if raw is None:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8", errors="replace")
        except Exception:
            return {}
    if not isinstance(raw, str):
        return {}
    s = raw.strip()
    if not s:
        return {}
    try:
        v = json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


def _collect_trait_keys(db) -> list[dict[str, Any]]:
    rows = db.execute(
        "SELECT id, result_json FROM black_barn_records WHERE status = 'done' ORDER BY id DESC LIMIT 500"
    ).fetchall()
    keys: dict[str, str] = {}
    for row in rows:
        rj = _bb_parse_result_json(row["result_json"])
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
        parsed = _bb_parse_result_json(row["result_json"])
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
