"""
GenMate Black Barn — inferência Holstein (lateral / posterior, imagem ou vídeo).

- Motor Perspicuus: `get_engine("black_barn")` (YOLO bb_yolo + ONNX lateral/posterior).
- Vídeo único: vista por campo no formulário ou heurística por nomes de classe do modelo
  de identificação (Ultralytics .pt / ONNX) quando configurado em `bb_identification`.
- Segmentação e pose (.pt Ultralytics): carregamento lazy; resultados guardados em `result_json`.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _view_from_class_name(name: str) -> Optional[str]:
    n = (name or "").strip().lower()
    if not n:
        return None
    if "posterior" in n or "post" in n or "traseir" in n:
        return "posterior"
    if "lateral" in n or "side" in n or "perfil" in n:
        return "lateral"
    return None


def _vote_views(views: List[str]) -> str:
    lat = sum(1 for v in views if v == "lateral")
    post = sum(1 for v in views if v == "posterior")
    if post > lat:
        return "posterior"
    if lat > 0:
        return "lateral"
    return "lateral"


def infer_view_with_identification_model(
    frame_bgr: np.ndarray,
    model_path: Optional[str],
) -> Optional[str]:
    """
    Usa modelo bb_identification (.pt YOLO ou ONNX export) na frame.
    Devolve 'lateral'|'posterior' conforme a classe da melhor deteção, ou None.
    """
    if not model_path or not os.path.isfile(model_path):
        return None
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".pt":
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            log.warning("[BlackBarn] ultralytics não instalado; não é possível usar .pt")
            return None
        try:
            m = YOLO(model_path)
            res = m.predict(frame_bgr, verbose=False)
            if not res or res[0].boxes is None or len(res[0].boxes) == 0:
                return None
            b = res[0].boxes[0]
            cid = int(b.cls[0])
            names = getattr(res[0], "names", None) or getattr(m, "names", {})
            label = str(names.get(cid, ""))
            return _view_from_class_name(label)
        except Exception:
            log.exception("[BlackBarn] identificação .pt falhou")
            return None
    if ext == ".onnx":
        try:
            import onnxruntime as ort  # type: ignore

            from perspicuus_inference import _create_ort_session, postprocess_yolo, letterbox, YOLO_INPUT_SIZE

            sess = _create_ort_session(model_path)
            inp = sess.get_inputs()[0].name
            is_fp16 = "float16" in sess.get_inputs()[0].type
            img, scale, pad_top, pad_left = letterbox(frame_bgr, YOLO_INPUT_SIZE)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            arr = rgb.transpose(2, 0, 1)[np.newaxis]
            if is_fp16:
                arr = arr.astype(np.float16)
            out = sess.run(None, {inp: arr})
            h, w = frame_bgr.shape[:2]
            dets = postprocess_yolo(out, h, w, scale, pad_top, pad_left)
            if not dets:
                return None
            _x1, _y1, _x2, _y2, _conf, cls_id = dets[0]
            if int(cls_id) == 1:
                return "posterior"
            return "lateral"
        except Exception:
            log.exception("[BlackBarn] identificação ONNX falhou")
            return None
    return None


def sample_video_frames(path: str, max_frames: int = 12) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        frames: List[np.ndarray] = []
        for _ in range(max_frames):
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            frames.append(fr)
        cap.release()
        return frames
    step = max(1, n // max_frames)
    frames: List[np.ndarray] = []
    for i in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok and fr is not None:
            frames.append(fr)
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames


def run_ultralytics_segmentation(
    frame_bgr: np.ndarray,
    model_path: Optional[str],
) -> Dict[str, Any]:
    if not model_path or not os.path.isfile(model_path):
        return {"ok": False, "error": "modelo_seg_em_falta"}
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        return {"ok": False, "error": "ultralytics_nao_instalado"}
    try:
        m = YOLO(model_path)
        r = m.predict(frame_bgr, verbose=False)[0]
        masks = getattr(r, "masks", None)
        if masks is None or masks.data is None:
            return {"ok": True, "n_masks": 0, "masks_shape": None}
        data = masks.data.cpu().numpy() if hasattr(masks.data, "cpu") else np.asarray(masks.data)
        return {"ok": True, "n_masks": int(data.shape[0]), "masks_shape": list(data.shape)}
    except Exception as e:
        log.exception("[BlackBarn] segmentação")
        return {"ok": False, "error": str(e)}


def run_ultralytics_pose(
    frame_bgr: np.ndarray,
    model_path: Optional[str],
) -> Dict[str, Any]:
    if not model_path or not os.path.isfile(model_path):
        return {"ok": False, "error": "modelo_pose_em_falta"}
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        return {"ok": False, "error": "ultralytics_nao_instalado"}
    try:
        m = YOLO(model_path)
        r = m.predict(frame_bgr, verbose=False)[0]
        k = getattr(r, "keypoints", None)
        if k is None or k.xy is None:
            return {"ok": True, "n_instances": 0}
        xy = k.xy.cpu().numpy() if hasattr(k.xy, "cpu") else np.asarray(k.xy)
        return {"ok": True, "n_instances": int(xy.shape[0]), "kpts_shape": list(xy.shape)}
    except Exception as e:
        log.exception("[BlackBarn] pose")
        return {"ok": False, "error": str(e)}


def _resolve_paths() -> Tuple[Any, Any, Any, Any]:
    from perspicuus_inference import get_engine as get_eng, resolve_model_path

    return (
        resolve_model_path("bb_identification"),
        resolve_model_path("bb_seg"),
        resolve_model_path("bb_pose"),
        get_eng,
    )


def process_record_on_disk(record_id: int, db_path: str, uploads_root: str) -> None:
    """Worker em thread: lê `black_barn_records`, corre pipeline, grava `result_json` / estado."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM black_barn_records WHERE id = ?", (record_id,)).fetchone()
    if not row:
        conn.close()
        return
    r = dict(row)
    ident_path, seg_path, pose_path, get_engine_fn = _resolve_paths()

    try:
        kind = (r.get("kind") or "image").strip()
        view_hint = (r.get("inferred_view") or "").strip().lower()

        def abs_path_disk(web: str) -> Optional[str]:
            if not web or not web.startswith("/api/black-barn/media/"):
                return None
            rest = web[len("/api/black-barn/media/") :].lstrip("/")
            if "/" not in rest:
                return None
            a, b = rest.split("/", 1)
            from werkzeug.utils import secure_filename

            ef, fn = secure_filename(a), secure_filename(b)
            fp = os.path.abspath(os.path.join(uploads_root, "black_barn", ef, fn))
            base = os.path.abspath(os.path.join(uploads_root, "black_barn", ef))
            if fp.startswith(base + os.sep) and os.path.isfile(fp):
                return fp
            return None

        result: Dict[str, Any] = {
            "kind": kind,
            "perspicuus": None,
            "segmentation": None,
            "pose": None,
            "views_voted": [],
        }

        eng = get_engine_fn("black_barn")

        if kind == "image":
            p = abs_path_disk(r.get("public_single") or "")
            if not p:
                raise ValueError("ficheiro_imagem_invalido")
            img = cv2.imread(p)
            if img is None:
                raise ValueError("cv2_imread_falhou")
            view = view_hint if view_hint in ("lateral", "posterior") else None
            if view is None and ident_path:
                view = infer_view_with_identification_model(img, ident_path)
            if view is None:
                view = "lateral"
            result["views_voted"] = [view]
            result["inferred_view_used"] = view
            if eng.is_ready():
                result["perspicuus"] = eng.infer_bgr(img, view)
            else:
                result["perspicuus"] = {"error": "motor_perspicuus_incompleto"}
            result["segmentation"] = run_ultralytics_segmentation(img, seg_path)
            result["pose"] = run_ultralytics_pose(img, pose_path)

        elif kind in ("video_single", "video_dual"):
            paths: List[str] = []
            if kind == "video_dual":
                for fld in ("public_lateral", "public_posterior"):
                    ap = abs_path_disk(r.get(fld) or "")
                    if ap:
                        paths.append(ap)
            else:
                ap = abs_path_disk(r.get("public_single") or "")
                if ap:
                    paths.append(ap)
            if not paths:
                raise ValueError("video_em_falta")
            all_views: List[str] = []
            persp_frames: List[Dict[str, Any]] = []
            seg_summ: List[Dict[str, Any]] = []
            pose_summ: List[Dict[str, Any]] = []
            for clip_i, vp in enumerate(paths):
                if kind == "video_dual":
                    clip_default = "lateral" if clip_i == 0 else "posterior"
                else:
                    clip_default = None
                frames = sample_video_frames(vp, max_frames=10)
                views_this: List[str] = []
                for fr in frames:
                    v = view_hint if view_hint in ("lateral", "posterior") else None
                    if v is None and clip_default is not None:
                        v = clip_default
                    if v is None and ident_path:
                        v = infer_view_with_identification_model(fr, ident_path)
                    if v is None:
                        v = clip_default or "lateral"
                    views_this.append(v)
                    if eng.is_ready():
                        persp_frames.append(eng.infer_bgr(fr, v))
                    seg_summ.append(run_ultralytics_segmentation(fr, seg_path))
                    pose_summ.append(run_ultralytics_pose(fr, pose_path))
                if views_this:
                    all_views.append(_vote_views(views_this))
            final_view = _vote_views(all_views) if all_views else "lateral"
            result["views_voted"] = all_views
            result["inferred_view_used"] = final_view
            result["perspicuus"] = {
                "aggregate": "mean_traits_pending",
                "frames": persp_frames[:24],
            }
            result["segmentation"] = {"per_clip": seg_summ[:8]}
            result["pose"] = {"per_clip": pose_summ[:8]}
        else:
            raise ValueError("kind_invalido")

        conn.execute(
            """
            UPDATE black_barn_records
            SET status = ?, result_json = ?, error_text = NULL, inferred_view = ?
            WHERE id = ?
            """,
            ("done", json.dumps(result, ensure_ascii=False), result.get("inferred_view_used") or "", record_id),
        )
        conn.commit()
    except Exception as e:
        log.exception("[BlackBarn] process_record id=%s", record_id)
        conn.execute(
            "UPDATE black_barn_records SET status = ?, error_text = ? WHERE id = ?",
            ("error", str(e), record_id),
        )
        conn.commit()
    finally:
        conn.close()
