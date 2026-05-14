"""
GenMate Black Barn — inferência Holstein (lateral / posterior, imagem ou vídeo).

- Motor Perspicuus: `get_engine("black_barn")` — o mesmo YOLO ONNX (`bb_yolo`) que no
  Perspicuus Brete/Holandês: deteção + crop e, em modelos multi-classe, a classe indica
  «lateral» ou «posterior» (nomes ou ids 0/1 como no export CowView).
- Voto automático de vista: usa `bb_identification` se existir; caso contrário reutiliza
  o ficheiro de `bb_yolo` (mesma inferência, sem segundo modelo obrigatório).
- Segmentação e pose (Ultralytics .pt ou export .onnx): lazy; resultados em `result_json`.
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


def _bb_yolo_max_det() -> int:
    """Limite de deteções por frame (1 = um animal). Ultralytics antigo pode não suportar o kw — há fallback."""
    raw = (os.environ.get("BB_YOLO_MAX_DET") or "1").strip()
    try:
        return max(1, min(300, int(raw)))
    except ValueError:
        return 1


def _bb_best_box_index(r: Any) -> int:
    """Índice da deteção com maior confiança da caixa; empate: maior área (segmentação/pose)."""
    boxes = getattr(r, "boxes", None)
    if boxes is None:
        return 0
    try:
        n = len(boxes)
    except TypeError:
        return 0
    if n <= 1:
        return 0
    try:
        conf_t = boxes.conf
        c = conf_t.cpu().numpy() if hasattr(conf_t, "cpu") else np.asarray(conf_t, dtype=np.float64)
        bxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy, dtype=np.float64)
        best_i = 0
        best_key = (-1.0, -1.0)
        for i in range(min(n, len(c), bxy.shape[0])):
            w = max(0.0, float(bxy[i, 2]) - float(bxy[i, 0]))
            h = max(0.0, float(bxy[i, 3]) - float(bxy[i, 1]))
            key = (float(c[i]), w * h)
            if key > best_key:
                best_key = key
                best_i = i
        return best_i
    except Exception:
        return 0


def _bb_best_segmentation_mask_index(r: Any) -> int:
    """Índice da máscara a manter (alinhada a `boxes` se contagens coincidirem; senão maior área)."""
    masks = getattr(r, "masks", None)
    # Nunca usar `not masks.xy`: em PyTorch `xy` pode ser Tensor — bool(tensor) é ambíguo.
    if masks is None or getattr(masks, "xy", None) is None:
        return _bb_best_box_index(r)
    try:
        n_m = len(masks.xy)
    except TypeError:
        return 0
    if n_m <= 1:
        return 0
    boxes = getattr(r, "boxes", None)
    if boxes is not None:
        try:
            if len(boxes) == n_m:
                return _bb_best_box_index(r)
        except TypeError:
            pass
    areas: List[float] = []
    for poly in masks.xy:
        if poly is None or len(poly) < 3:
            areas.append(0.0)
            continue
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        areas.append(float(cv2.contourArea(pts)))
    return int(np.argmax(np.asarray(areas, dtype=np.float64))) if areas else 0


def _bb_best_pose_person_index(r: Any) -> int:
    """Índice da pessoa/animal com maior confiança média nos keypoints (fallback: conf da caixa)."""
    k = getattr(r, "keypoints", None)
    if k is None or getattr(k, "xy", None) is None:
        return 0
    try:
        xy = k.xy.cpu().numpy() if hasattr(k.xy, "cpu") else np.asarray(k.xy, dtype=np.float32)
    except Exception:
        return 0
    n = int(xy.shape[0])
    if n <= 1:
        return 0
    sc = np.zeros(n, dtype=np.float64)
    kc = getattr(k, "conf", None)
    if kc is not None:
        try:
            arr = kc.cpu().numpy() if hasattr(kc, "cpu") else np.asarray(kc, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] == n:
                sc = np.mean(arr, axis=1)
            elif arr.ndim == 1 and len(arr) == n:
                sc = arr.astype(np.float64)
        except Exception:
            pass
    if float(np.max(sc)) <= 0.0:
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            try:
                nb = len(boxes)
                if nb >= n:
                    conf_t = boxes.conf
                    bc = conf_t.cpu().numpy() if hasattr(conf_t, "cpu") else np.asarray(conf_t, dtype=np.float64)
                    for i in range(n):
                        sc[i] = max(sc[i], float(bc[i]))
            except Exception:
                pass
    bi = int(np.argmax(sc))
    return bi


def _bb_seg_multi_instance(r: Any) -> bool:
    masks = getattr(r, "masks", None)
    n_m = 0
    if masks is not None and getattr(masks, "xy", None) is not None:
        try:
            n_m = len(masks.xy)
        except TypeError:
            n_m = 0
    boxes = getattr(r, "boxes", None)
    try:
        n_b = len(boxes) if boxes is not None else 0
    except TypeError:
        n_b = 0
    return max(n_m, n_b) > 1


def _bb_pose_multi_instance(r: Any) -> bool:
    k = getattr(r, "keypoints", None)
    if k is None or getattr(k, "xy", None) is None:
        return False
    try:
        xy = k.xy.cpu().numpy() if hasattr(k.xy, "cpu") else np.asarray(k.xy, dtype=np.float32)
        return int(xy.shape[0]) > 1
    except Exception:
        return False


def _bb_boxes_xyxy_numpy(r: Any) -> Optional[np.ndarray]:
    boxes = getattr(r, "boxes", None)
    if boxes is None or not getattr(boxes, "xyxy", None):
        return None
    try:
        bxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy, dtype=np.float64)
        if bxy.size == 0 or bxy.ndim != 2 or bxy.shape[1] < 4:
            return None
        return bxy
    except Exception:
        return None


def _bb_mask_polygon_centroid(poly: Any) -> Optional[Tuple[float, float]]:
    if poly is None or len(poly) < 3:
        return None
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def _bb_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return float(inter / ua) if ua > 1e-9 else 0.0


def _bb_point_in_xyxy(px: float, py: float, row: np.ndarray, pad_frac: float = 0.12) -> bool:
    x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
    return (x1 - pad_frac * w) <= px <= (x2 + pad_frac * w) and (y1 - pad_frac * h) <= py <= (y2 + pad_frac * h)


def _bb_individual_segmentation_mask_indices(r: Any) -> List[int]:
    """
    Índices de máscara do mesmo indivíduo: ancora na melhor bbox (confiança + área).
    Inclui outras deteções cujo centro de caixa está dentro da bbox âncora ou com IoU alto
    (várias classes / partes do mesmo animal). Se #máscaras ≠ #caixas, usa centroide do polígono.
    """
    masks = getattr(r, "masks", None)
    if masks is None or getattr(masks, "xy", None) is None:
        return []
    try:
        n_m = len(masks.xy)
    except TypeError:
        return []
    if n_m <= 0:
        return []
    if n_m == 1:
        return [0]

    bxy = _bb_boxes_xyxy_numpy(r)
    if bxy is None or int(bxy.shape[0]) == 0:
        return [_bb_best_segmentation_mask_index(r)]

    n_b = int(bxy.shape[0])
    bi = _bb_best_box_index(r) if n_b > 1 else 0
    bi = max(0, min(bi, n_b - 1))
    anchor = bxy[bi]
    kept: set[int] = set()

    if n_b == n_m:
        for j in range(n_b):
            if j == bi:
                kept.add(j)
                continue
            cxb = 0.5 * (float(bxy[j, 0]) + float(bxy[j, 2]))
            cyb = 0.5 * (float(bxy[j, 1]) + float(bxy[j, 3]))
            inside = _bb_point_in_xyxy(cxb, cyb, anchor, pad_frac=0.15)
            ov = _bb_iou_xyxy(bxy[j], anchor)
            if inside or ov >= 0.25:
                kept.add(j)
    else:
        for m in range(n_m):
            poly = masks.xy[m]
            cen = _bb_mask_polygon_centroid(poly)
            if cen and _bb_point_in_xyxy(cen[0], cen[1], anchor, pad_frac=0.12):
                kept.add(m)
        if not kept:
            acx = 0.5 * (float(anchor[0]) + float(anchor[2]))
            acy = 0.5 * (float(anchor[1]) + float(anchor[3]))
            best_m = -1
            best_d = float("inf")
            for m in range(n_m):
                cen = _bb_mask_polygon_centroid(masks.xy[m])
                if not cen:
                    continue
                d = (cen[0] - acx) ** 2 + (cen[1] - acy) ** 2
                if d < best_d:
                    best_d, best_m = d, m
            if best_m >= 0:
                kept.add(best_m)
            else:
                kept.add(_bb_best_segmentation_mask_index(r))

    out = sorted(kept)
    if not out:
        out = [min(bi, n_m - 1)]
    return out


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
            try:
                res = m.predict(frame_bgr, verbose=False, max_det=_bb_yolo_max_det())
            except TypeError:
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
    kw: Dict[str, Any] = {
        "verbose": False,
        "half": False,
        "retina_masks": False,
        "max_det": _bb_yolo_max_det(),
    }
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
            kw.pop("max_det", None)
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
                    try:
                        r = m.predict(  # type: ignore[misc]
                            [im],
                            verbose=False,
                            half=False,
                            retina_masks=False,
                            max_det=_bb_yolo_max_det(),
                        )[0]
                    except TypeError:
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


def _mask_tensor_from_ultralytics(masks_obj: Any) -> Any:
    """Extrai tensor/array de máscaras; evita OrderedDict e estruturas aninhadas sem .float()."""
    try:
        import torch  # type: ignore
    except ImportError:
        torch = None  # type: ignore

    if masks_obj is None:
        return None

    def _coalesce(x: Any) -> Any:
        if x is None:
            return None
        if torch is not None and isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, dict):
            for v in x.values():
                got = _coalesce(v)
                if got is not None:
                    return got
        return None

    for attr in ("data", "masks"):
        raw = getattr(masks_obj, attr, None)
        t = _coalesce(raw)
        if t is not None:
            return t
    return None


def _tensor_to_numpy_u8_rgb(arr: Any) -> Optional[np.ndarray]:
    """Normaliza saída de Results.plot() para array uint8 3 canais (BGR, compatível cv2.imencode)."""
    if arr is None:
        return None
    try:
        x = np.asarray(arr)
    except Exception:
        return None
    if x.size == 0:
        return None
    if x.dtype == object:
        return None
    if x.dtype != np.uint8:
        mx = float(np.nanmax(x)) if x.size else 0.0
        if mx <= 1.01:
            x = (np.clip(x, 0, 1) * 255.0).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)
    if x.ndim == 2:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    elif x.ndim == 3 and x.shape[2] == 4:
        x = x[:, :, :3]
    elif x.ndim != 3 or x.shape[2] != 3:
        return None
    return x


def _bb_segmentation_overlay_numpy(frame_bgr: np.ndarray, r: Any) -> np.ndarray:
    """Fallback sem Results.plot() — uma máscara + uma caixa (melhor confiança / área)."""
    out = frame_bgr.copy()
    masks = getattr(r, "masks", None)
    if masks is not None and getattr(masks, "xy", None) is not None:
        try:
            n_m = len(masks.xy)
        except TypeError:
            n_m = 0
        mids = _bb_individual_segmentation_mask_indices(r) if n_m > 0 else []
        overlay = out.copy()
        col = (40, 200, 80)
        for mi in mids:
            if not (0 <= mi < n_m):
                continue
            poly = masks.xy[mi]
            if poly is not None and len(poly) >= 3:
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts_i], col, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.42, out, 0.58, 0, dst=out)
    boxes = getattr(r, "boxes", None)
    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
        try:
            bxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            n_b = int(bxy.shape[0])
            bi = _bb_best_box_index(r) if n_b > 1 else 0
            if n_b > 0 and 0 <= bi < n_b:
                row = bxy[bi]
                x1, y1, x2, y2 = [int(round(float(v))) for v in row[:4]]
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2, cv2.LINE_AA)
        except Exception:
            pass
    return out


def _bb_pose_overlay_numpy(frame_bgr: np.ndarray, r: Any) -> np.ndarray:
    """Fallback sem Results.plot() — keypoints de um único indivíduo (BGR)."""
    out = frame_bgr.copy()
    k = getattr(r, "keypoints", None)
    if k is None or getattr(k, "xy", None) is None:
        return out
    try:
        xy = k.xy
        pts = xy.cpu().numpy() if hasattr(xy, "cpu") else np.asarray(xy, dtype=np.float32)
    except Exception:
        return out
    n_p = int(pts.shape[0])
    bi = _bb_best_pose_person_index(r) if n_p > 1 else 0
    if 0 <= bi < n_p:
        inst = pts[bi]
        for j in range(inst.shape[0]):
            x, y = float(inst[j, 0]), float(inst[j, 1])
            if x < 0.5 or y < 0.5:
                continue
            cv2.circle(out, (int(round(x)), int(round(y))), 4, (0, 220, 255), -1, cv2.LINE_AA)
    return out


def _bb_pose_draw_keypoint_indices(img: np.ndarray, r: Any) -> np.ndarray:
    """Escreve o índice de cada keypoint no plot (um indivíduo)."""
    k = getattr(r, "keypoints", None)
    if k is None or getattr(k, "xy", None) is None:
        return img
    try:
        xy = k.xy
        pts = xy.cpu().numpy() if hasattr(xy, "cpu") else np.asarray(xy, dtype=np.float32)
    except Exception:
        return img
    out = np.ascontiguousarray(img)
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(max(0.38, min(0.75, w / 1000.0)))
    thick = max(1, int(round(scale * 2)))
    n_p = int(pts.shape[0])
    bi = _bb_best_pose_person_index(r) if n_p > 1 else 0
    if not (0 <= bi < n_p):
        return out
    for j in range(pts.shape[1]):
        x, y = float(pts[bi, j, 0]), float(pts[bi, j, 1])
        if x < 0.5 or y < 0.5:
            continue
        xi, yi = int(round(x)), int(round(y))
        label = str(j)
        tx = min(w - 4, max(4, xi + 5))
        ty = max(16, yi - 5)
        cv2.putText(out, label, (tx, ty), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(out, label, (tx, ty), font, scale, (0, 255, 255), thick, cv2.LINE_AA)
    return out


def bb_render_segmentation_plot_png(model_path: str, frame_bgr: np.ndarray) -> bytes:
    from ultralytics import YOLO  # type: ignore

    m = YOLO(model_path)
    r = _bb_yolo_predict_one_result(m, frame_bgr)
    arr_u8: Optional[np.ndarray] = None
    if _bb_seg_multi_instance(r):
        arr_u8 = _bb_segmentation_overlay_numpy(frame_bgr, r)
    else:
        try:
            plotted = r.plot()
            arr_u8 = _tensor_to_numpy_u8_rgb(plotted)
        except Exception as ex:  # noqa: BLE001 — ex.: OrderedDict sem .float() dentro do Ultralytics
            log.warning("[BlackBarn] seg r.plot(): %s", ex)
    if arr_u8 is None or arr_u8.size == 0:
        arr_u8 = _bb_segmentation_overlay_numpy(frame_bgr, r)
    ok, buf = cv2.imencode(".png", arr_u8)
    return bytes(buf) if ok else bb_png_message_bytes("segmentação: PNG falhou")


def bb_render_pose_plot_png(model_path: str, frame_bgr: np.ndarray) -> bytes:
    from ultralytics import YOLO  # type: ignore

    m = YOLO(model_path)
    r = _bb_yolo_predict_one_result(m, frame_bgr)
    arr_u8: Optional[np.ndarray] = None
    if _bb_pose_multi_instance(r):
        arr_u8 = _bb_pose_overlay_numpy(frame_bgr, r)
    else:
        try:
            plotted = r.plot()
            arr_u8 = _tensor_to_numpy_u8_rgb(plotted)
        except Exception as ex:  # noqa: BLE001
            log.warning("[BlackBarn] pose r.plot(): %s", ex)
    if arr_u8 is None or arr_u8.size == 0:
        arr_u8 = _bb_pose_overlay_numpy(frame_bgr, r)
    arr_u8 = _bb_pose_draw_keypoint_indices(arr_u8, r)
    ok, buf = cv2.imencode(".png", arr_u8)
    return bytes(buf) if ok else bb_png_message_bytes("pose: PNG falhou")


def _bb_read_yaml_dict_from_ultralytics_model(m: Any) -> Optional[Dict[str, Any]]:
    """Tenta obter o dict `yaml` embutido no modelo Ultralytics (Pose / Detection)."""
    inner = getattr(m, "model", None)
    if inner is not None:
        y = getattr(inner, "yaml", None)
        if isinstance(y, dict):
            return y
    y = getattr(m, "yaml", None)
    if isinstance(y, dict):
        return y
    return None


def _bb_kpt_names_list_from_yaml_dict(y: Dict[str, Any]) -> Optional[List[str]]:
    """Extrai lista de nomes de keypoints do campo `kpt_names` (formato Ultralytics datasets)."""
    kn = y.get("kpt_names")
    if kn is None:
        return None
    if isinstance(kn, dict):
        lst = kn.get(0)
        if lst is None:
            lst = kn.get("0")
        if lst is None and kn:
            lst = kn.get(next(iter(kn.keys())))
        if isinstance(lst, (list, tuple)):
            return [str(x).strip() for x in lst if str(x).strip()]
        return None
    if isinstance(kn, (list, tuple)):
        return [str(x).strip() for x in kn if str(x).strip()]
    return None


def _bb_load_yaml_file_dict(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _bb_resolve_kpt_names_from_train_data_paths(model: Any) -> Optional[List[str]]:
    """Lê `data=*.yaml` de overrides ou checkpoint (treino) para obter `kpt_names`."""
    candidates: List[str] = []
    ov = getattr(model, "overrides", None)
    if isinstance(ov, dict) and isinstance(ov.get("data"), str):
        candidates.append(ov["data"])
    ck = getattr(model, "ckpt", None)
    if isinstance(ck, dict):
        ta = ck.get("train_args") or {}
        if isinstance(ta, dict) and isinstance(ta.get("data"), str):
            candidates.append(ta["data"])
    for p in candidates:
        yd = _bb_load_yaml_file_dict(p)
        if yd is None and not os.path.isabs(p):
            yd = _bb_load_yaml_file_dict(os.path.join(os.getcwd(), p))
        if not yd:
            continue
        names = _bb_kpt_names_list_from_yaml_dict(yd)
        if names:
            return names
    return None


# Ordem COCO-pose oficial (apenas fallback se o .pt não trouxer `kpt_names` e nk==17)
_KP_COCO_DEFAULT17: Tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def bb_resolve_pose_keypoint_names(model: Any, n_kpts: int) -> List[str]:
    """
    Identifica nomes de keypoints a partir do metadata do modelo Ultralytics:
    `model.yaml` → `kpt_names`, ou ficheiro `data` do treino; senão `kp{j}` ou COCO17 genérico.
    """
    nk = max(0, int(n_kpts))
    raw: Optional[List[str]] = None
    y = _bb_read_yaml_dict_from_ultralytics_model(model)
    if isinstance(y, dict):
        raw = _bb_kpt_names_list_from_yaml_dict(y)
        if raw is None and isinstance(y.get("data"), str):
            dp = y["data"]
            yd = _bb_load_yaml_file_dict(dp)
            if yd is None and not os.path.isabs(dp):
                yd = _bb_load_yaml_file_dict(os.path.join(os.getcwd(), dp))
            if isinstance(yd, dict):
                raw = _bb_kpt_names_list_from_yaml_dict(yd)
    if raw is None:
        raw = _bb_resolve_kpt_names_from_train_data_paths(model)
    if raw and len(raw) >= nk:
        return list(raw[:nk])
    if raw and len(raw) > 0:
        return list(raw) + [f"kp{i}" for i in range(len(raw), nk)]
    if nk == len(_KP_COCO_DEFAULT17):
        return list(_KP_COCO_DEFAULT17)
    return [f"kp{i}" for i in range(nk)]


def _bb_segmentation_instances_from_result(frame_bgr: np.ndarray, r: Any) -> Dict[str, Any]:
    h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    diag = float((w * w + h * h) ** 0.5) or 1.0
    inst: List[Dict[str, Any]] = []
    masks = getattr(r, "masks", None)
    if masks is not None and getattr(masks, "xy", None) is not None:
        try:
            n_m = len(masks.xy)
        except TypeError:
            n_m = 0
        for mid in _bb_individual_segmentation_mask_indices(r):
            if not (0 <= mid < n_m):
                continue
            poly = masks.xy[mid]
            if poly is None or len(poly) < 3:
                continue
            pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
            x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            area = float(cv2.contourArea(pts.astype(np.float32)))
            inst.append(
                {
                    "id": int(mid),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "centroid": [cx, cy],
                    "area_px": max(0.0, area),
                    "width": max(0.0, x2 - x1),
                    "height": max(0.0, y2 - y1),
                }
            )
    if not inst:
        boxes = getattr(r, "boxes", None)
        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
            try:
                bxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
                n_b = int(bxy.shape[0])
                bi = _bb_best_box_index(r) if n_b > 1 else 0
                if n_b > 0 and 0 <= bi < n_b:
                    row = bxy[bi]
                    x1, y1, x2, y2 = [float(v) for v in row[:4]]
                    inst.append(
                        {
                            "id": 0,
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "centroid": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                            "area_px": max(0.0, (x2 - x1) * (y2 - y1)),
                            "width": max(0.0, x2 - x1),
                            "height": max(0.0, y2 - y1),
                        }
                    )
            except Exception:
                pass
    return {"ok": True, "width": w, "height": h, "diag": diag, "n_instances": len(inst), "instances": inst}


def bb_segmentation_instances_json_with_model(m: Any, frame_bgr: np.ndarray) -> Dict[str, Any]:
    """Mesmo que `bb_segmentation_instances_json`, mas reutiliza instância YOLO já carregada."""
    try:
        r = _bb_yolo_predict_one_result(m, frame_bgr)
        return _bb_segmentation_instances_from_result(frame_bgr, r)
    except Exception as e:  # noqa: BLE001
        log.exception("[BlackBarn] seg geometry (modelo reutilizado)")
        return {"ok": False, "error": str(e)[:200]}


def bb_segmentation_instances_json(model_path: str, frame_bgr: np.ndarray) -> Dict[str, Any]:
    """Geometria por instância de segmentação (polígonos / bbox) para UI e traits. Modelo: .pt ou .onnx (Ultralytics)."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        return {"ok": False, "error": "ultralytics_nao_instalado"}
    if not model_path or not os.path.isfile(model_path):
        return {"ok": False, "error": "modelo_seg_em_falta"}
    try:
        m = YOLO(model_path)
        r = _bb_yolo_predict_one_result(m, frame_bgr)
        return _bb_segmentation_instances_from_result(frame_bgr, r)
    except Exception as e:  # noqa: BLE001
        log.exception("[BlackBarn] seg geometry")
        return {"ok": False, "error": str(e)[:200]}


def _bb_pose_keypoints_from_result(model: Any, frame_bgr: np.ndarray, r: Any) -> Dict[str, Any]:
    h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    diag = float((w * w + h * h) ** 0.5) or 1.0
    k = getattr(r, "keypoints", None)
    if k is None or getattr(k, "xy", None) is None:
        return {"ok": True, "width": w, "height": h, "diag": diag, "n_instances": 0, "keypoint_names": [], "instances": []}
    xy = k.xy.cpu().numpy() if hasattr(k.xy, "cpu") else np.asarray(k.xy, dtype=np.float32)
    conf = None
    if getattr(k, "conf", None) is not None:
        try:
            kc = k.conf
            conf = kc.cpu().numpy() if hasattr(kc, "cpu") else np.asarray(kc, dtype=np.float32)
        except Exception:
            conf = None
    nk = int(xy.shape[1])
    names = bb_resolve_pose_keypoint_names(model, nk)
    instances: List[Dict[str, Any]] = []
    n_p = int(xy.shape[0])
    bi = _bb_best_pose_person_index(r) if n_p > 1 else 0
    if 0 <= bi < n_p:
        kps: List[Dict[str, Any]] = []
        for j in range(nk):
            vx, vy = float(xy[bi, j, 0]), float(xy[bi, j, 1])
            cj = None
            if conf is not None:
                if conf.ndim == 2 and bi < conf.shape[0] and j < conf.shape[1]:
                    cj = float(conf[bi, j])
                elif conf.ndim == 1 and j < conf.shape[0]:
                    cj = float(conf[j])
            kps.append({"i": j, "name": names[j], "x": vx, "y": vy, "conf": cj})
        instances.append({"id": 0, "keypoints": kps})
    return {
        "ok": True,
        "width": w,
        "height": h,
        "diag": diag,
        "n_instances": len(instances),
        "keypoint_names": names,
        "instances": instances,
    }


def bb_pose_keypoints_json_with_model(m: Any, frame_bgr: np.ndarray) -> Dict[str, Any]:
    """Mesmo que `bb_pose_keypoints_json`, mas reutiliza instância YOLO já carregada."""
    try:
        r = _bb_yolo_predict_one_result(m, frame_bgr)
        return _bb_pose_keypoints_from_result(m, frame_bgr, r)
    except Exception as e:  # noqa: BLE001
        log.exception("[BlackBarn] pose geometry (modelo reutilizado)")
        return {"ok": False, "error": str(e)[:200]}


def bb_pose_keypoints_json(model_path: str, frame_bgr: np.ndarray) -> Dict[str, Any]:
    """Keypoints por instância (coordenadas + confiança) para UI e traits. Modelo: .pt ou .onnx (Ultralytics)."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        return {"ok": False, "error": "ultralytics_nao_instalado"}
    if not model_path or not os.path.isfile(model_path):
        return {"ok": False, "error": "modelo_pose_em_falta"}
    try:
        m = YOLO(model_path)
        r = _bb_yolo_predict_one_result(m, frame_bgr)
        return _bb_pose_keypoints_from_result(m, frame_bgr, r)
    except Exception as e:  # noqa: BLE001
        log.exception("[BlackBarn] pose geometry")
        return {"ok": False, "error": str(e)[:200]}


def bb_trait_value_from_seg_geom(data: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[float]:
    """Calcula valor de trait de segmentação a partir do JSON de geometria (alinhado à UI)."""
    import math

    if not data or not data.get("ok"):
        return None
    metric = str(cfg.get("metric") or "")
    raw_ids = cfg.get("mask_indices") or cfg.get("mask_ids") or []
    try:
        ids_int = sorted(int(x) for x in raw_ids)
    except (TypeError, ValueError):
        return None
    inst_list = data.get("instances") or []
    W = float(data.get("width") or 1) or 1.0
    H = float(data.get("height") or 1) or 1.0
    D = float(data.get("diag") or 1) or 1.0

    def by_id(iid: int) -> Optional[Dict[str, Any]]:
        for ins in inst_list:
            if int(ins.get("id", -999999)) == int(iid):
                return ins if isinstance(ins, dict) else None
        return None

    if metric == "mask_centroid_distance_norm_diag":
        if len(ids_int) < 2:
            return None
        a, b = by_id(ids_int[0]), by_id(ids_int[1])
        if not a or not b:
            return None
        try:
            ax, ay = float(a["centroid"][0]), float(a["centroid"][1])
            bx, by = float(b["centroid"][0]), float(b["centroid"][1])
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        return float(math.hypot(ax - bx, ay - by) / D)
    if metric == "mask_area_ratio":
        if len(ids_int) < 2:
            return None
        a, b = by_id(ids_int[0]), by_id(ids_int[1])
        if not a or not b:
            return None
        denom = float(b.get("area_px") or 0)
        if denom <= 0:
            return None
        return float(a.get("area_px") or 0) / denom
    if metric == "mask_width_norm_w":
        if len(ids_int) < 1:
            return None
        a = by_id(ids_int[0])
        if not a:
            return None
        return float(a.get("width") or 0) / W
    if metric == "mask_height_norm_h":
        if len(ids_int) < 1:
            return None
        a = by_id(ids_int[0])
        if not a:
            return None
        return float(a.get("height") or 0) / H
    return None


def bb_trait_value_from_kp_geom(data: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[float]:
    """Calcula valor de trait de pose a partir do JSON de keypoints (instância 0 por defeito na UI)."""
    import math

    if not data or not data.get("ok"):
        return None
    metric = str(cfg.get("metric") or "")
    raw_ids = cfg.get("kp_indices") or []
    try:
        ids_int = sorted(int(x) for x in raw_ids)
    except (TypeError, ValueError):
        return None
    inst_idx = int(cfg.get("instance", 0) or 0)
    instances = data.get("instances") or []
    if inst_idx < 0 or inst_idx >= len(instances):
        return None
    inst0 = instances[inst_idx]
    if not isinstance(inst0, dict):
        return None
    D = float(data.get("diag") or 1) or 1.0

    def pt(ki: int) -> Optional[Tuple[float, float]]:
        for kp in inst0.get("keypoints") or []:
            if not isinstance(kp, dict):
                continue
            if int(kp.get("i", -1)) == int(ki):
                try:
                    return float(kp["x"]), float(kp["y"])
                except (KeyError, TypeError, ValueError):
                    return None
        return None

    if metric == "kp_distance_norm_diag":
        if len(ids_int) < 2:
            return None
        a, b = pt(ids_int[0]), pt(ids_int[1])
        if not a or not b:
            return None
        return float(math.hypot(a[0] - b[0], a[1] - b[1]) / D)
    if metric == "kp_angle_norm_pi":
        if len(ids_int) < 3:
            return None
        i0, i1, i2 = ids_int[0], ids_int[1], ids_int[2]
        p0, p1, p2 = pt(i0), pt(i1), pt(i2)
        if not p0 or not p1 or not p2:
            return None
        ax, ay = p0[0] - p1[0], p0[1] - p1[1]
        bx, by = p2[0] - p1[0], p2[1] - p1[1]
        na = math.hypot(ax, ay)
        nb = math.hypot(bx, by)
        if na <= 0 or nb <= 0:
            return None
        c = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
        return float(math.acos(c) / math.pi)
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
        if k is None or getattr(k, "xy", None) is None:
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


def _bb_gc_soft() -> None:
    try:
        import gc

        gc.collect()
    except Exception:
        pass


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
            _bb_gc_soft()
            result["views_voted"] = [view]
            result["inferred_view_used"] = view
            if eng.is_ready():
                result["perspicuus"] = eng.infer_bgr(img, view)
            else:
                result["perspicuus"] = {"error": "motor_perspicuus_incompleto"}
            _bb_gc_soft()
            result["segmentation"] = run_ultralytics_segmentation(img, seg_path)
            _bb_gc_soft()
            result["pose"] = run_ultralytics_pose(img, pose_path)
            _bb_gc_soft()

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
