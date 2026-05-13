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
from typing import Any, Dict, List, Optional

from flask import abort, flash, jsonify, redirect, render_template, request, send_from_directory, url_for, Response
from werkzeug.utils import secure_filename

log = logging.getLogger(__name__)

BB_REPROCESS_JOBS: Dict[str, Dict[str, Any]] = {}
BB_REPROCESS_LOCK = threading.Lock()

BB_TRAIT_APPLY_JOBS: Dict[str, Dict[str, Any]] = {}
BB_TRAIT_APPLY_LOCK = threading.Lock()

# Caminho DB definido em register_black_barn — usado para persistir jobs entre workers Gunicorn.
BB_TRAIT_JOB_DB_PATH: Optional[str] = None
BB_TRAIT_JOB_LAST_PERSIST_TS: Dict[str, float] = {}
BB_TRAIT_JOB_PERSIST_MIN_S = 0.35


def _bb_ensure_trait_jobs_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS black_barn_trait_jobs (
            job_id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def _bb_persist_trait_apply_job(job_id: str, *, force: bool = False) -> None:
    """Grava estado do job na BD para GET /status noutro worker Gunicorn."""
    import time

    if not BB_TRAIT_JOB_DB_PATH:
        return
    if not force:
        nowt = time.monotonic()
        with BB_TRAIT_APPLY_LOCK:
            last = BB_TRAIT_JOB_LAST_PERSIST_TS.get(job_id, 0.0)
            if nowt - last < BB_TRAIT_JOB_PERSIST_MIN_S:
                return
            BB_TRAIT_JOB_LAST_PERSIST_TS[job_id] = nowt
    with BB_TRAIT_APPLY_LOCK:
        job = BB_TRAIT_APPLY_JOBS.get(job_id)
    if not job:
        return
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        conn = sqlite3.connect(BB_TRAIT_JOB_DB_PATH, timeout=60)
        try:
            _bb_ensure_trait_jobs_table(conn)
            conn.execute(
                """
                INSERT INTO black_barn_trait_jobs (job_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (job_id, json.dumps(job, ensure_ascii=False, default=str), now),
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as e:
        log.warning("[BlackBarn] persist trait job %s: %s", job_id, e)


def _bb_load_trait_apply_job_from_db(db_path: str, job_id: str) -> Optional[Dict[str, Any]]:
    try:
        conn = sqlite3.connect(db_path, timeout=20)
        try:
            _bb_ensure_trait_jobs_table(conn)
            row = conn.execute(
                "SELECT payload_json FROM black_barn_trait_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        finally:
            conn.close()
        if not row or row[0] is None:
            return None
        raw = row[0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return json.loads(raw)
    except Exception as e:  # noqa: BLE001
        log.warning("[BlackBarn] load trait job %s: %s", job_id, e)
        return None


def _bb_init_trait_apply_job(
    *,
    trait_key: str,
    source: str,
    farm_id: str,
    per_frame: bool,
    total_records: int,
    started_by: str,
) -> str:
    job_id = uuid.uuid4().hex
    with BB_TRAIT_APPLY_LOCK:
        BB_TRAIT_APPLY_JOBS[job_id] = {
            "job_id": job_id,
            "mode": "single",
            "trait_key": trait_key,
            "source": source,
            "farm_id_filter": farm_id or None,
            "per_frame": bool(per_frame),
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "finished_at": None,
            "started_by": started_by,
            "total_records": int(total_records),
            "processed_records": 0,
            "inserted": 0,
            "skipped": 0,
            "errors": 0,
            "current_record_id": None,
            "current_step": "A calcular traits em todos os registos",
            "sample_errors": [],
            "trait_failures": [],
        }
    _bb_persist_trait_apply_job(job_id, force=True)
    return job_id


def _bb_update_trait_apply_job(job_id: str, **fields: Any) -> None:
    with BB_TRAIT_APPLY_LOCK:
        job = BB_TRAIT_APPLY_JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
    force = fields.get("status") in ("done", "failed") or fields.get("finished_at") is not None
    _bb_persist_trait_apply_job(job_id, force=bool(force))


def _bb_init_trait_recalc_all_job(*, source: str, farm_id: str, started_by: str) -> str:
    job_id = uuid.uuid4().hex
    with BB_TRAIT_APPLY_LOCK:
        BB_TRAIT_APPLY_JOBS[job_id] = {
            "job_id": job_id,
            "mode": "recalc_all_defs",
            "trait_key": None,
            "source": source,
            "farm_id_filter": farm_id or None,
            "per_frame": False,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "finished_at": None,
            "started_by": started_by,
            "traits_total": 0,
            "trait_index": 0,
            "current_trait_key": None,
            "total_records": 0,
            "processed_records": 0,
            "inserted": 0,
            "skipped": 0,
            "errors": 0,
            "current_record_id": None,
            "current_step": "A preparar recálculo de todos os traits",
            "sample_errors": [],
            "trait_failures": [],
        }
    _bb_persist_trait_apply_job(job_id, force=True)
    return job_id


def _bb_apply_one_trait_to_records(
    job_id: str,
    conn: sqlite3.Connection,
    m: Any,
    trait_key: str,
    source: str,
    per_frame: bool,
    cfg: Dict[str, Any],
    rows: List[sqlite3.Row],
    uploads_dir: str,
) -> tuple[int, int, int]:
    """Apaga valores antigos deste trait (mesmo source), recalcula para todos os `rows`."""
    from black_barn_inference import (
        bb_load_frame_bgr_for_record,
        bb_pose_keypoints_json_with_model,
        bb_preview_frame_count_for_record,
        bb_segmentation_instances_json_with_model,
        bb_trait_value_from_kp_geom,
        bb_trait_value_from_seg_geom,
    )

    conn.execute(
        """
        DELETE FROM black_barn_trait_values WHERE trait_key = ? AND EXISTS (
            SELECT 1 FROM black_barn_trait_defs d
            WHERE d.trait_key = black_barn_trait_values.trait_key AND d.source = ?
        )
        """,
        (trait_key, source),
    )
    conn.commit()

    ins_total = 0
    skip_total = 0
    err_total = 0
    total = len(rows)
    with BB_TRAIT_APPLY_LOCK:
        j = BB_TRAIT_APPLY_JOBS.get(job_id)
        single_mode = j is not None and j.get("mode") == "single"
    if single_mode:
        _bb_update_trait_apply_job(job_id, total_records=total)

    if total == 0:
        return 0, 0, 0

    for i, row in enumerate(rows, start=1):
        rid = int(row["id"])
        rdict = dict(row)
        _bb_update_trait_apply_job(
            job_id,
            processed_records=i - 1,
            current_record_id=rid,
            current_step=f"{trait_key} · registo {rid} ({i}/{total})",
        )
        nfc = bb_preview_frame_count_for_record(rdict, uploads_dir, "auto")
        if nfc <= 0:
            skip_total += 1
            _bb_update_trait_apply_job(job_id, processed_records=i, skipped=skip_total, inserted=ins_total)
            continue
        frame_iter = range(nfc) if per_frame else (0,)
        try:
            for fi in frame_iter:
                img = bb_load_frame_bgr_for_record(rdict, uploads_dir, fi, "auto")
                if img is None:
                    skip_total += 1
                    continue
                if source == "seg":
                    data = bb_segmentation_instances_json_with_model(m, img)
                    v = bb_trait_value_from_seg_geom(data, cfg)
                else:
                    data = bb_pose_keypoints_json_with_model(m, img)
                    v = bb_trait_value_from_kp_geom(data, cfg)
                if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    skip_total += 1
                    continue
                fi_sql = int(fi) if per_frame else None
                conn.execute(
                    """
                    INSERT INTO black_barn_trait_values (record_id, trait_key, frame_index, value)
                    VALUES (?, ?, ?, ?)
                    """,
                    (rid, trait_key, fi_sql, float(v)),
                )
                ins_total += 1
            conn.commit()
        except Exception as ex:  # noqa: BLE001
            log.exception("[BlackBarn] trait apply rid=%s trait=%s", rid, trait_key)
            err_total += 1
            conn.rollback()
            se_new: Optional[List[Dict[str, Any]]] = None
            with BB_TRAIT_APPLY_LOCK:
                job = BB_TRAIT_APPLY_JOBS.get(job_id)
                if job and len(job.get("sample_errors") or []) < 25:
                    se_new = list(job.get("sample_errors") or [])
                    se_new.append({"id": rid, "trait_key": trait_key, "error": str(ex)})
            if se_new is not None:
                _bb_update_trait_apply_job(job_id, sample_errors=se_new)
        _bb_update_trait_apply_job(
            job_id,
            processed_records=i,
            inserted=ins_total,
            skipped=skip_total,
            errors=err_total,
        )

    return ins_total, skip_total, err_total


def _bb_run_trait_apply_job(
    job_id: str,
    trait_key: str,
    source: str,
    farm_id: str,
    per_frame: bool,
    db_path: str,
    uploads_dir: str,
) -> None:
    from perspicuus_inference import resolve_model_path, reset_engine

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ins_total = 0
    skip_total = 0
    err_total = 0
    try:
        reset_engine("black_barn")
        row_def = conn.execute(
            """
            SELECT trait_key, label, source, config_json FROM black_barn_trait_defs
            WHERE trait_key = ? AND source = ? AND farm_id = ?
            """,
            (trait_key, source, farm_id or "default"),
        ).fetchone()
        if not row_def:
            row_def = conn.execute(
                """
                SELECT trait_key, label, source, config_json FROM black_barn_trait_defs
                WHERE trait_key = ? AND source = ?
                ORDER BY id DESC LIMIT 1
                """,
                (trait_key, source),
            ).fetchone()
        if not row_def:
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            _bb_update_trait_apply_job(
                job_id,
                status="failed",
                finished_at=now,
                current_step="Definição de trait não encontrada na BD",
            )
            return
        try:
            cfg = json.loads(row_def["config_json"] or "{}")
        except (json.JSONDecodeError, TypeError, ValueError):
            cfg = {}
        effective_per_frame = bool(per_frame) or bool(cfg.get("per_frame"))
        if source == "seg":
            model_path = resolve_model_path("bb_seg")
        else:
            model_path = resolve_model_path("bb_pose")
        if not model_path or not os.path.isfile(model_path):
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            _bb_update_trait_apply_job(
                job_id,
                status="failed",
                finished_at=now,
                current_step="Modelo .pt não configurado (bb_seg ou bb_pose)",
            )
            return

        max_n = max(1, min(10000, int(os.environ.get("BB_TRAIT_APPLY_MAX", "5000"))))
        if farm_id:
            rows = conn.execute(
                "SELECT * FROM black_barn_records WHERE farm_id = ? ORDER BY id ASC LIMIT ?",
                (farm_id, max_n),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM black_barn_records ORDER BY id ASC LIMIT ?",
                (max_n,),
            ).fetchall()
        total = len(rows)
        _bb_update_trait_apply_job(job_id, total_records=total)
        if total == 0:
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            _bb_update_trait_apply_job(
                job_id,
                status="done",
                finished_at=now,
                current_step="Sem registos Black Barn na BD",
                current_record_id=None,
                processed_records=0,
                inserted=0,
                skipped=0,
                errors=0,
            )
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            _bb_update_trait_apply_job(
                job_id,
                status="failed",
                finished_at=now,
                current_step="ultralytics não instalado",
            )
            return

        m = YOLO(model_path)
        ins_total, skip_total, err_total = _bb_apply_one_trait_to_records(
            job_id,
            conn,
            m,
            trait_key,
            source,
            effective_per_frame,
            cfg,
            rows,
            uploads_dir,
        )

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        _bb_update_trait_apply_job(
            job_id,
            status="done",
            finished_at=now,
            current_step="Concluído",
            current_record_id=None,
            processed_records=total,
            inserted=ins_total,
            skipped=skip_total,
            errors=err_total,
        )
    except Exception as e:  # noqa: BLE001
        log.exception("[BlackBarn] trait apply job=%s", job_id)
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with BB_TRAIT_APPLY_LOCK:
            j = BB_TRAIT_APPLY_JOBS.get(job_id) or {}
            pr = int(j.get("processed_records") or 0)
        _bb_update_trait_apply_job(
            job_id,
            status="failed",
            finished_at=now,
            current_step=str(e)[:220],
            processed_records=pr,
            inserted=ins_total,
            skipped=skip_total,
            errors=err_total,
        )
    finally:
        conn.close()


def _bb_run_recalc_all_traits_job(
    job_id: str,
    source: str,
    farm_id: str,
    db_path: str,
    uploads_dir: str,
) -> None:
    """Recalcula todas as definições de trait (seg ou kp) para todos os animais/registos."""
    from perspicuus_inference import resolve_model_path, reset_engine

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ins_all = skip_all = err_all = 0
    try:
        reset_engine("black_barn")
        defs = conn.execute(
            """
            SELECT id, farm_id, trait_key, label, source, config_json
            FROM black_barn_trait_defs
            WHERE source = ?
            ORDER BY id ASC
            """,
            (source,),
        ).fetchall()
        if not defs:
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            _bb_update_trait_apply_job(
                job_id,
                status="done",
                finished_at=now,
                current_step="Nenhuma definição de trait para este tipo",
                traits_total=0,
                trait_index=0,
                inserted=0,
                skipped=0,
                errors=0,
            )
            return

        if source == "seg":
            model_path = resolve_model_path("bb_seg")
        else:
            model_path = resolve_model_path("bb_pose")
        if not model_path or not os.path.isfile(model_path):
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            _bb_update_trait_apply_job(
                job_id,
                status="failed",
                finished_at=now,
                current_step="Modelo .pt não configurado (bb_seg ou bb_pose)",
            )
            return

        max_n = max(1, min(10000, int(os.environ.get("BB_TRAIT_APPLY_MAX", "5000"))))
        if farm_id:
            rows = conn.execute(
                "SELECT * FROM black_barn_records WHERE farm_id = ? ORDER BY id ASC LIMIT ?",
                (farm_id, max_n),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM black_barn_records ORDER BY id ASC LIMIT ?",
                (max_n,),
            ).fetchall()
        nrows = len(rows)
        nt = len(defs)
        _bb_update_trait_apply_job(job_id, traits_total=nt, total_records=nrows)

        if nrows == 0:
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            _bb_update_trait_apply_job(
                job_id,
                status="done",
                finished_at=now,
                current_step="Sem registos Black Barn na BD",
                traits_total=nt,
                inserted=0,
                skipped=0,
                errors=0,
            )
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            _bb_update_trait_apply_job(
                job_id,
                status="failed",
                finished_at=now,
                current_step="ultralytics não instalado",
            )
            return

        m = YOLO(model_path)
        trait_failures: List[Dict[str, Any]] = []
        for di, drow in enumerate(defs):
            tk = str(drow["trait_key"])
            try:
                cfg = json.loads(drow["config_json"] or "{}")
            except (json.JSONDecodeError, TypeError, ValueError):
                cfg = {}
            effective_per_frame = bool(cfg.get("per_frame"))
            _bb_update_trait_apply_job(
                job_id,
                trait_index=di,
                current_trait_key=tk,
                processed_records=0,
                current_step=f"Recalcular {tk} ({di + 1}/{nt})",
                trait_failures=list(trait_failures),
            )
            try:
                ins, ski, err = _bb_apply_one_trait_to_records(
                    job_id, conn, m, tk, source, effective_per_frame, cfg, rows, uploads_dir
                )
                ins_all += ins
                skip_all += ski
                err_all += err
            except Exception as ex:  # noqa: BLE001
                log.exception("[BlackBarn] recalc trait=%s job=%s", tk, job_id)
                trait_failures.append({"trait_key": tk, "error": str(ex)[:400]})
                err_all += 1
                _bb_update_trait_apply_job(
                    job_id,
                    trait_failures=list(trait_failures),
                    current_step=f"Erro em {tk}; a continuar ({di + 1}/{nt})",
                )

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        step_done = (
            "Concluído (todos os traits)"
            if not trait_failures
            else f"Concluído com {len(trait_failures)} trait(s) em falha — ver trait_failures"
        )
        _bb_update_trait_apply_job(
            job_id,
            status="done",
            finished_at=now,
            current_step=step_done,
            current_record_id=None,
            trait_index=max(0, nt - 1),
            processed_records=nrows,
            inserted=ins_all,
            skipped=skip_all,
            errors=err_all,
            trait_failures=trait_failures,
        )
    except Exception as e:  # noqa: BLE001
        log.exception("[BlackBarn] recalc all traits job=%s", job_id)
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with BB_TRAIT_APPLY_LOCK:
            j = BB_TRAIT_APPLY_JOBS.get(job_id) or {}
            pr = int(j.get("processed_records") or 0)
        _bb_update_trait_apply_job(
            job_id,
            status="failed",
            finished_at=now,
            current_step=str(e)[:220],
            processed_records=pr,
            inserted=ins_all,
            skipped=skip_all,
            errors=err_all,
        )
    finally:
        conn.close()


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


def _bb_delete_record_media_files(uploads_root: str, rec: Dict[str, Any]) -> None:
    """Apaga ficheiros de media do registo; só caminhos dentro de uploads_root/black_barn/."""
    base = os.path.abspath(os.path.join(uploads_root, "black_barn"))
    for key in ("path_single", "path_lateral", "path_posterior"):
        raw = rec.get(key)
        if not raw or not isinstance(raw, str):
            continue
        p = os.path.abspath(raw.strip())
        if not p.startswith(base + os.sep):
            log.warning("[BlackBarn] recusado apagar ficheiro fora de black_barn: %s", p)
            continue
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError as ex:
            log.warning("[BlackBarn] não apagou %s: %s", p, ex)


def register_black_barn(app) -> None:
    import app as main

    global BB_TRAIT_JOB_DB_PATH
    BB_TRAIT_JOB_DB_PATH = main.DB_PATH

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

    @app.route("/api/genmate-black-barn/records/<int:rid>", methods=["DELETE"])
    @main.login_required
    def api_black_barn_record_delete(rid: int):
        db = main.get_db()
        row = db.execute("SELECT * FROM black_barn_records WHERE id = ?", (rid,)).fetchone()
        if not row:
            return jsonify({"error": "not_found"}), 404
        rec = dict(row)
        _bb_delete_record_media_files(main.UPLOADS_DIR, rec)
        db.execute("DELETE FROM black_barn_records WHERE id = ?", (rid,))
        db.commit()
        return jsonify({"ok": True, "id": rid})

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
            seg_data_base=url_for("api_black_barn_preview_segmentation_data", rid=0).rsplit("/", 1)[0] + "/",
            macro_rows=_bb_macro_stats(db, "seg"),
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
            pose_data_base=url_for("api_black_barn_preview_pose_data", rid=0).rsplit("/", 1)[0] + "/",
            macro_rows=_bb_macro_stats(db, "kp"),
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
        from black_barn_inference import bb_preview_frame_count_for_record

        pfc = bb_preview_frame_count_for_record(r, main.UPLOADS_DIR, "auto")
        seg_preview_url = url_for("api_black_barn_preview_segmentation", rid=rid)
        pose_preview_url = url_for("api_black_barn_preview_pose", rid=rid)
        return render_template(
            "black_barn_individual.html",
            record=r,
            trait_rows=[dict(x) for x in traits],
            seg_preview_url=seg_preview_url,
            pose_preview_url=pose_preview_url,
            preview_frame_count=pfc,
        )

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

    @app.route("/api/genmate-black-barn/preview/segmentation-data/<int:rid>", methods=["GET"])
    @main.login_required
    def api_black_barn_preview_segmentation_data(rid: int):
        from perspicuus_inference import resolve_model_path

        from black_barn_inference import (
            bb_load_frame_bgr_for_record,
            bb_segmentation_instances_json,
        )

        frame = int(request.args.get("frame", 0) or 0)
        clip = str(request.args.get("clip", "auto") or "auto").strip().lower()
        if clip not in ("auto", "single", "lateral", "posterior"):
            clip = "auto"
        db = main.get_db()
        row = db.execute("SELECT * FROM black_barn_records WHERE id = ?", (rid,)).fetchone()
        if not row:
            return jsonify({"ok": False, "error": "registo não encontrado"}), 404
        rec = dict(row)
        img = bb_load_frame_bgr_for_record(rec, main.UPLOADS_DIR, frame, clip)
        if img is None:
            return jsonify({"ok": False, "error": "sem mídia ou frame inválido"}), 400
        seg = resolve_model_path("bb_seg")
        if not seg or not os.path.isfile(seg):
            return jsonify({"ok": False, "error": "modelo bb_seg não configurado"}), 400
        data = bb_segmentation_instances_json(seg, img)
        data["record_id"] = rid
        data["frame"] = frame
        data["clip"] = clip
        return jsonify(data)

    @app.route("/api/genmate-black-barn/preview/pose-data/<int:rid>", methods=["GET"])
    @main.login_required
    def api_black_barn_preview_pose_data(rid: int):
        from perspicuus_inference import resolve_model_path

        from black_barn_inference import bb_load_frame_bgr_for_record, bb_pose_keypoints_json

        frame = int(request.args.get("frame", 0) or 0)
        clip = str(request.args.get("clip", "auto") or "auto").strip().lower()
        if clip not in ("auto", "single", "lateral", "posterior"):
            clip = "auto"
        db = main.get_db()
        row = db.execute("SELECT * FROM black_barn_records WHERE id = ?", (rid,)).fetchone()
        if not row:
            return jsonify({"ok": False, "error": "registo não encontrado"}), 404
        rec = dict(row)
        img = bb_load_frame_bgr_for_record(rec, main.UPLOADS_DIR, frame, clip)
        if img is None:
            return jsonify({"ok": False, "error": "sem mídia ou frame inválido"}), 400
        pose = resolve_model_path("bb_pose")
        if not pose or not os.path.isfile(pose):
            return jsonify({"ok": False, "error": "modelo bb_pose não configurado"}), 400
        data = bb_pose_keypoints_json(pose, img)
        data["record_id"] = rid
        data["frame"] = frame
        data["clip"] = clip
        return jsonify(data)

    @app.route("/api/genmate-black-barn/correlations/points")
    @main.login_required
    def api_black_barn_correlation_points():
        xk = request.args.get("x", "").strip()
        yk = request.args.get("y", "").strip()
        if not xk or not yk or xk == yk:
            return jsonify({
                "error": "parâmetros_x_y",
                "n": 0,
                "points": [],
                "pearson": None,
                "spearman": None,
                "kendall": None,
            }), 400
        db = main.get_db()
        pts, nx, ny = _paired_trait_points(db, xk, yk)
        label_x = _bb_correlation_axis_label(db, xk)
        label_y = _bb_correlation_axis_label(db, yk)
        if len(pts) < 2:
            return jsonify({
                "n": len(pts),
                "points": pts,
                "pearson": None,
                "spearman": None,
                "kendall": None,
                "label_x": label_x,
                "label_y": label_y,
            })
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
            "label_x": label_x,
            "label_y": label_y,
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
                ON CONFLICT(farm_id, trait_key) DO UPDATE SET
                    label = excluded.label,
                    source = excluded.source,
                    config_json = excluded.config_json,
                    created_at = excluded.created_at
                """,
                (now, farm_id, trait_key, label, source, cfg),
            )
            db.commit()
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"ok": True})

    @app.route("/api/genmate-black-barn/trait-apply-all/start", methods=["POST"])
    @main.login_required
    def api_black_barn_trait_apply_all_start():
        data = request.get_json(silent=True) or {}
        trait_key = str(data.get("trait_key") or "").strip()[:80]
        source = str(data.get("source") or "seg").strip()[:16]
        if source not in ("seg", "kp"):
            return jsonify({"error": "source"}), 400
        if not trait_key:
            return jsonify({"error": "trait_key"}), 400
        farm_id = secure_filename(str(data.get("farm_id") or "").strip() or "")
        per_frame = bool(data.get("per_frame"))
        job_id = _bb_init_trait_apply_job(
            trait_key=trait_key,
            source=source,
            farm_id=farm_id,
            per_frame=per_frame,
            total_records=0,
            started_by=str(main.session.get("username") or "unknown"),
        )
        threading.Thread(
            target=_bb_run_trait_apply_job,
            args=(job_id, trait_key, source, farm_id, per_frame, main.DB_PATH, main.UPLOADS_DIR),
            daemon=True,
        ).start()
        return jsonify({"status": "started", "job_id": job_id})

    @app.route("/api/genmate-black-barn/trait-apply-all/status/<job_id>", methods=["GET"])
    @main.login_required
    def api_black_barn_trait_apply_all_status(job_id: str):
        with BB_TRAIT_APPLY_LOCK:
            job = dict(BB_TRAIT_APPLY_JOBS.get(job_id) or {})
        if not job:
            loaded = _bb_load_trait_apply_job_from_db(main.DB_PATH, job_id)
            if loaded:
                job = dict(loaded)
        if not job:
            return jsonify({"error": "job_not_found"}), 404
        mode = job.get("mode") or "single"
        if mode == "recalc_all_defs":
            tt = int(job.get("traits_total") or 0)
            tr = int(job.get("total_records") or 0)
            ti = int(job.get("trait_index") or 0)
            pr = int(job.get("processed_records") or 0)
            if tt <= 0 or tr <= 0:
                pct = 100 if job.get("status") == "done" else 0
            else:
                denom = tt * tr
                numer = ti * tr + min(max(pr, 0), tr)
                pct = int(round((numer / denom) * 100)) if denom > 0 else 0
        else:
            total = int(job.get("total_records") or 0)
            processed = int(job.get("processed_records") or 0)
            pct = int(round((processed / total) * 100)) if total > 0 else 0
        job["progress_pct"] = max(0, min(100, pct))
        return jsonify(job)

    @app.route("/api/genmate-black-barn/trait-recalc-all/start", methods=["POST"])
    @main.login_required
    def api_black_barn_trait_recalc_all_start():
        data = request.get_json(silent=True) or {}
        source = str(data.get("source") or "seg").strip()[:16]
        if source not in ("seg", "kp"):
            return jsonify({"error": "source"}), 400
        farm_id = secure_filename(str(data.get("farm_id") or "").strip() or "")
        job_id = _bb_init_trait_recalc_all_job(
            source=source,
            farm_id=farm_id,
            started_by=str(main.session.get("username") or "unknown"),
        )
        threading.Thread(
            target=_bb_run_recalc_all_traits_job,
            args=(job_id, source, farm_id, main.DB_PATH, main.UPLOADS_DIR),
            daemon=True,
        ).start()
        return jsonify({"status": "started", "job_id": job_id})

    @app.route("/api/genmate-black-barn/trait-defs/<int:def_id>", methods=["PATCH", "DELETE"])
    @main.login_required
    def api_black_barn_trait_def_one(def_id: int):
        db = main.get_db()
        row = db.execute(
            "SELECT id, farm_id, trait_key, label, source, config_json FROM black_barn_trait_defs WHERE id = ?",
            (def_id,),
        ).fetchone()
        if not row:
            return jsonify({"error": "not_found"}), 404
        src = str(row["source"] or "")
        if src not in ("seg", "kp"):
            return jsonify({"error": "source_nao_editavel"}), 400
        if request.method == "DELETE":
            tk = str(row["trait_key"])
            db.execute("DELETE FROM black_barn_trait_values WHERE trait_key = ?", (tk,))
            db.execute("DELETE FROM black_barn_trait_defs WHERE id = ?", (def_id,))
            db.commit()
            return jsonify({"ok": True})
        data = request.get_json(silent=True) or {}
        label = data.get("label")
        config = data.get("config")
        if label is None and config is None:
            return jsonify({"error": "label_ou_config"}), 400
        new_label = str(row["label"] or "")
        if label is not None:
            new_label = str(label).strip()[:160] or new_label
        cfg_raw = row["config_json"] or "{}"
        try:
            cur_cfg = json.loads(cfg_raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            cur_cfg = {}
        if config is not None:
            if not isinstance(config, dict):
                return jsonify({"error": "config_invalido"}), 400
            cur_cfg = config
        cfg_out = json.dumps(cur_cfg, ensure_ascii=False)
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        db.execute(
            "UPDATE black_barn_trait_defs SET label = ?, config_json = ?, created_at = ? WHERE id = ?",
            (new_label, cfg_out, now, def_id),
        )
        db.commit()
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


def _bb_macro_stats(db, source: str) -> List[Dict[str, Any]]:
    """Agregados por trait (segmentação `seg` ou keypoints `kp`) para visão macro."""
    if source not in ("seg", "kp"):
        return []
    rows = db.execute(
        """
        SELECT tv.trait_key AS trait_key,
               COUNT(*) AS n,
               AVG(tv.value) AS mean_v,
               MIN(tv.value) AS min_v,
               MAX(tv.value) AS max_v
        FROM black_barn_trait_values tv
        INNER JOIN black_barn_trait_defs d ON d.trait_key = tv.trait_key AND d.source = ?
        GROUP BY tv.trait_key
        ORDER BY tv.trait_key
        LIMIT 48
        """,
        (source,),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        tk = row["trait_key"]
        lr = db.execute(
            "SELECT label FROM black_barn_trait_defs WHERE trait_key = ? AND source = ? LIMIT 1",
            (tk, source),
        ).fetchone()
        label = lr["label"] if lr and lr["label"] else tk
        top = db.execute(
            """
            SELECT r.id AS id, r.animal_tag AS animal_tag, tv.value AS value
            FROM black_barn_trait_values tv
            JOIN black_barn_records r ON r.id = tv.record_id
            WHERE tv.trait_key = ?
            ORDER BY tv.value DESC
            LIMIT 5
            """,
            (tk,),
        ).fetchall()
        rank_rows = db.execute(
            """
            SELECT r.id AS id, r.animal_tag AS animal_tag, tv.value AS value, tv.frame_index AS fi
            FROM black_barn_trait_values tv
            JOIN black_barn_records r ON r.id = tv.record_id
            INNER JOIN black_barn_trait_defs d ON d.trait_key = tv.trait_key AND d.source = ?
            WHERE tv.trait_key = ?
            ORDER BY tv.value DESC, r.id ASC, tv.frame_index ASC
            LIMIT 500
            """,
            (source, tk),
        ).fetchall()
        ranking = [
            {
                "rank": rank_i,
                "id": int(t["id"]),
                "animal_tag": t["animal_tag"],
                "value": round(float(t["value"]), 6),
                "frame_index": t["fi"],
            }
            for rank_i, t in enumerate(rank_rows, start=1)
        ]
        vals = [
            float(x["value"])
            for x in db.execute(
                """
                SELECT tv.value FROM black_barn_trait_values tv
                INNER JOIN black_barn_trait_defs d ON d.trait_key = tv.trait_key AND d.source = ?
                WHERE tv.trait_key = ?
                """,
                (source, tk),
            ).fetchall()
        ]
        std_v: float | None = None
        if len(vals) > 1:
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            std_v = round(var**0.5, 6)
        out.append(
            {
                "trait_key": tk,
                "label": label,
                "n": int(row["n"] or 0),
                "mean": round(float(row["mean_v"]), 6) if row["mean_v"] is not None else None,
                "std": std_v,
                "min": round(float(row["min_v"]), 6) if row["min_v"] is not None else None,
                "max": round(float(row["max_v"]), 6) if row["max_v"] is not None else None,
                "top": [
                    {"id": int(t["id"]), "animal_tag": t["animal_tag"], "value": round(float(t["value"]), 6)}
                    for t in top
                ],
                "ranking": ranking,
            }
        )
    return out


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


def _bb_correlation_axis_label(db, key: str) -> str:
    """Rótulo legível para eixos do scatter (Perspicuus ou trait custom na BD)."""
    if key.startswith("persp:"):
        return key.replace("persp:", "Perspicuus · ", 1)
    if key.startswith("custom:"):
        tk = key.split(":", 1)[1]
        row = db.execute(
            "SELECT label FROM black_barn_trait_defs WHERE trait_key = ? LIMIT 1",
            (tk,),
        ).fetchone()
        if row and row["label"]:
            return str(row["label"])
        return tk
    return key


def _paired_trait_points(db, xk: str, yk: str) -> tuple[list[list[float]], str, str]:
    """Extrai pares (x,y) por registo onde ambas as medidas existem."""
    # Incluir todos os registos recentes: traits custom gravados na BD existem mesmo que
    # `status` ainda não seja `done`; valores por frame usam `frame_index` não nulo.
    rows = db.execute(
        "SELECT id, result_json FROM black_barn_records ORDER BY id DESC LIMIT 500"
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
        """
        SELECT AVG(value) AS v FROM black_barn_trait_values
        WHERE record_id = ? AND trait_key = ?
        """,
        (record_id, tname),
    ).fetchone()
    if not row or row["v"] is None:
        return None
    try:
        v = float(row["v"])
        return v if v == v else None
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
