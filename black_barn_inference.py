"""
GenMate Black Barn — inferência Holstein (lateral / posterior, imagem ou vídeo).

- Motor Perspicuus: `get_engine("black_barn")` — o mesmo YOLO ONNX (`bb_yolo`) que no
  Perspicuus Brete/Holandês: deteção + crop e, em modelos multi-classe, a classe indica
  «lateral» ou «posterior» (nomes ou ids 0/1 como no export CowView).
- Voto automático de vista: usa `bb_identification` se existir; caso contrário reutiliza
  o ficheiro de `bb_yolo` (mesma inferência, sem segundo modelo obrigatório).
- Segmentação e pose (.pt Ultralytics): lazy; resultados em `result_json`.
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

BB_VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv")
BB_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


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


def bb_identification_model_path() -> Optional[str]:
    """
    Caminho para votar lateral/posterior: slot `bb_identification`, senão o mesmo
    ONNX que `bb_yolo` (comportamento Perspicuus — um YOLO faz crop e identifica a vista).
    """
    from perspicuus_inference import resolve_model_path

    for role in ("bb_identification", "bb_yolo"):
        p = resolve_model_path(role)
        if p and os.path.isfile(p):
            return p
    return None


def infer_view_with_identification_model(
    frame_bgr: np.ndarray,
    model_path: Optional[str],
) -> Optional[str]:
    """
    Corre o YOLO de vista na frame (.pt Ultralytics ou ONNX, típico export CowView).
    Devolve 'lateral'|'posterior' pela classe da melhor deteção (nomes ou id 0/1), ou None.
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


def _ultralytics_predict_first(
    m: Any,
    frame_bgr: np.ndarray,
) -> Any:
    """
    predict() com opções estáveis em CPU (evita caminhos retina / half que em algumas
    versões disparam erros internos, ex. OrderedDict sem .float()).
    """
    import numpy as _np

    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("frame vazio")
    img = _np.ascontiguousarray(frame_bgr)
    if img.dtype != _np.uint8:
        img = _np.clip(img, 0, 255).astype(_np.uint8)
    kw: Dict[str, Any] = {"verbose": False, "half": False, "retina_masks": False}
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            kw["device"] = "cpu"
    except Exception:
        pass
    try:
        return m.predict(img, **kw)[0]
    except TypeError:
        kw.pop("device", None)
        try:
            return m.predict(img, **kw)[0]
        except TypeError:
            return m.predict(img, verbose=False)[0]


def bb_media_disk_path(web_path: str, uploads_root: str) -> Optional[str]:
    """Converte URL `/api/black-barn/media/...` em caminho absoluto no disco."""
    if not web_path or not str(web_path).startswith("/api/black-barn/media/"):
        return None
    rest = str(web_path)[len("/api/black-barn/media/") :].lstrip("/")
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


def bb_pick_media_disk(record: Dict[str, Any], uploads_root: str, clip: str = "auto") -> Optional[str]:
    clip = (clip or "auto").strip().lower()
    keys: List[str]
    if clip == "single":
        keys = ["public_single"]
    elif clip == "lateral":
        keys = ["public_lateral", "public_single"]
    elif clip == "posterior":
        keys = ["public_posterior", "public_single"]
    else:
        keys = ["public_single", "public_lateral", "public_posterior"]
    for k in keys:
        disk = bb_media_disk_path(str(record.get(k) or ""), uploads_root)
        if disk:
            return disk
    return None


def bb_video_frame_count_disk(disk_path: str) -> int:
    ext = os.path.splitext(disk_path)[1].lower()
    if ext not in BB_VIDEO_EXTS:
        return 0
    cap = cv2.VideoCapture(disk_path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return max(0, n)


def bb_preview_frame_count_for_record(record: Dict[str, Any], uploads_root: str, clip: str = "auto") -> int:
    disk = bb_pick_media_disk(record, uploads_root, clip)
    if not disk:
        return 0
    ext = os.path.splitext(disk)[1].lower()
    if ext in BB_IMAGE_EXTS:
        return 1
    n = bb_video_frame_count_disk(disk)
    return max(1, n) if n > 0 else 0


def bb_load_frame_bgr_for_record(
    record: Dict[str, Any],
    uploads_root: str,
    frame_index: int = 0,
    clip: str = "auto",
) -> Optional[np.ndarray]:
    disk = bb_pick_media_disk(record, uploads_root, clip)
    if not disk:
        return None
    ext = os.path.splitext(disk)[1].lower()
    if ext in BB_IMAGE_EXTS:
        return cv2.imread(disk)
    if ext in BB_VIDEO_EXTS:
        cap = cv2.VideoCapture(disk)
        if not cap.isOpened():
            return None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fi = max(0, min(int(frame_index), max(0, n - 1))) if n > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        cap.release()
        return fr if ok else None
    return None


def _bb_yolo_predict_one_result(m: Any, frame_bgr: np.ndarray) -> Any:
    """Primeiro `Results` Ultralytics (BGR/RGB + fallback lista)."""
    r = None
    errs: List[str] = []
    for im in (
        np.ascontiguousarray(frame_bgr),
        np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)),
    ):
        if im.dtype != np.uint8:
            im = np.clip(im, 0, 255).astype(np.uint8)
        for use_list in (False, True):
            try:
                if use_list:
                    r = m.predict(  # type: ignore[misc]
                        [im], verbose=False, half=False, retina_masks=False
                    )[0]
                else:
                    r = _ultralytics_predict_first(m, im)
                break
            except Exception as ex:  # noqa: BLE001
                errs.append(str(ex))
                r = None
        if r is not None:
            break
    if r is None:
        raise RuntimeError("; ".join(errs[-5:]) if errs else "predict falhou")
    return r


def bb_png_message_bytes(message: str) -> bytes:
    msg = (message or "?")[:200]
    w = max(420, min(1200, 12 * len(msg)))
    img = np.full((140, w, 3), 235, np.uint8)
    y = 50
    for line in [msg[i : i + 70] for i in range(0, len(msg), 70)] or [msg]:
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        y += 22
    ok, buf = cv2.imencode(".png", img)
    return bytes(buf) if ok else b""


def bb_render_segmentation_plot_png(model_path: str, frame_bgr: np.ndarray) -> bytes:
    from ultralytics import YOLO  # type: ignore

    m = YOLO(model_path)
    r = _bb_yolo_predict_one_result(m, frame_bgr)
    plotted = r.plot()
    if plotted is None:
        return bb_png_message_bytes("segmentação: plot vazio")
    arr = np.asarray(plotted)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    return bytes(buf) if ok else bb_png_message_bytes("segmentação: PNG falhou")


def bb_render_pose_plot_png(model_path: str, frame_bgr: np.ndarray) -> bytes:
    from ultralytics import YOLO  # type: ignore

    m = YOLO(model_path)
    r = _bb_yolo_predict_one_result(m, frame_bgr)
    plotted = r.plot()
    if plotted is None:
        return bb_png_message_bytes("pose: plot vazio")
    arr = np.asarray(plotted)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    return bytes(buf) if ok else bb_png_message_bytes("pose: PNG falhou")
    """Tensor de máscaras [N,H,W] ou None (sem tocar em atributos que disparam bugs)."""
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    if masks_obj is None:
        return None
    for attr in ("masks", "data"):
        raw = getattr(masks_obj, attr, None)
        if isinstance(raw, torch.Tensor):
            return raw
        if isinstance(raw, dict):
            for v in raw.values():
                if isinstance(v, torch.Tensor):
                    return v
    return None


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
        r = _bb_yolo_predict_one_result(m, frame_bgr)

        masks = getattr(r, "masks", None)
        shape_list: Optional[List[int]] = None
        n_masks = 0

        if masks is not None:
            t = _mask_tensor_from_ultralytics(masks)
            if t is not None:
                try:
                    if hasattr(t, "detach"):
                        arr = t.detach().float().cpu().numpy()
                    else:
                        arr = np.asarray(t, dtype=np.float32)
                    n_masks = int(arr.shape[0])
                    shape_list = list(arr.shape)
                except Exception:
                    n_masks = 0
            if n_masks == 0 and getattr(masks, "xy", None) is not None:
                try:
                    n_masks = len(masks.xy)
                except TypeError:
                    n_masks = 0
        if n_masks == 0 and getattr(r, "boxes", None) is not None:
            try:
                n_boxes = len(r.boxes)
                if n_boxes > 0 and masks is not None:
                    n_masks = n_boxes
            except TypeError:
                pass

        return {"ok": True, "n_masks": int(n_masks), "masks_shape": shape_list}
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
        r = _bb_yolo_predict_one_result(m, frame_bgr)
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
        bb_identification_model_path(),
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
