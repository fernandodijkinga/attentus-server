"""
Inferência Perspicuus MK1 (YOLO CowView + iudicium ONNX por vista).

Apenas vistas **lateral** e **posterior** — modelos distintos via variáveis de ambiente.
Armazena todas as traits devolvidas pelo ONNX (nomes vindos de metadata.json).

Variáveis de ambiente (paths absolutos ou relativos ao CWD) — têm prioridade sobre ficheiros em DATA_DIR/ml_models/:
  PERSPICUUS_YOLO_ONNX          — CowView / NeloreView (obrigatório para inferir)
  PERSPICUUS_LATERAL_ONNX       — modelo iudicium para lateral
  PERSPICUUS_POSTERIOR_ONNX     — modelo iudicium para posterior
  PERSPICUUS_LATERAL_METADATA_JSON   — opcional (trait_names, input_size, …)
  PERSPICUUS_POSTERIOR_METADATA_JSON — opcional

Sem env: usa registry.json em DATA_DIR/ml_models/ (upload pela UI / API interna).

Dependências: onnxruntime, opencv-python-headless, numpy, Pillow
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from werkzeug.utils import secure_filename

log = logging.getLogger(__name__)

# ─── Defaults alinhados a Perspicuus-2_mk1_inference-ECC.py ─────────────────
YOLO_INPUT_SIZE = 640
YOLO_CONF_THRES = 0.40
YOLO_IOU_THRES = 0.45
YOLO_MAX_DETECTIONS = 10
CROP_PAD = 0.05
BBOX_SHRINK = 1.0
TRAIT_NAMES_DEFAULT = ["T1"]
IMG_SIZE_DEFAULT = 224
SCORE_CLAMP = 5.0

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ─── Modelos em disco (upload UI) ───────────────────────────────────────────
ROLE_TO_ENV: Dict[str, str] = {
    "yolo": "PERSPICUUS_YOLO_ONNX",
    "lateral": "PERSPICUUS_LATERAL_ONNX",
    "posterior": "PERSPICUUS_POSTERIOR_ONNX",
    "lateral_meta": "PERSPICUUS_LATERAL_METADATA_JSON",
    "posterior_meta": "PERSPICUUS_POSTERIOR_METADATA_JSON",
}
REGISTRY_FILENAME = "registry.json"


def data_dir_default() -> str:
    return os.environ.get(
        "DATA_DIR",
        "/data" if os.environ.get("RENDER") else os.path.join(os.path.dirname(__file__), "data"),
    )


def get_models_dir() -> str:
    d = os.path.join(data_dir_default(), "ml_models")
    os.makedirs(d, exist_ok=True)
    return d


def load_registry() -> Dict[str, Optional[str]]:
    path = os.path.join(get_models_dir(), REGISTRY_FILENAME)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Optional[str]] = {}
    for k in ROLE_TO_ENV:
        v = data.get(k)
        if v is None or v == "":
            out[k] = None
        else:
            out[k] = str(v)
    return out


def save_registry(updates: Dict[str, Optional[str]]) -> None:
    base = load_registry()
    for k, v in updates.items():
        if k in ROLE_TO_ENV:
            base[k] = v
    path = os.path.join(get_models_dir(), REGISTRY_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2, ensure_ascii=False)
        f.write("\n")


def resolve_model_path(role: str) -> Optional[str]:
    """Env (ficheiro existente) > ficheiro em ml_models/registry."""
    if role not in ROLE_TO_ENV:
        return None
    env_p = os.environ.get(ROLE_TO_ENV[role], "").strip()
    if env_p and os.path.isfile(env_p):
        return env_p
    reg = load_registry()
    fn = reg.get(role)
    if not fn:
        return None
    safe = os.path.basename(str(fn))
    fp = os.path.join(get_models_dir(), safe)
    if os.path.isfile(fp):
        return fp
    return None


def model_path_source(role: str) -> str:
    """Para UI: origem do path efetivo."""
    if role not in ROLE_TO_ENV:
        return "missing"
    env_p = os.environ.get(ROLE_TO_ENV[role], "").strip()
    if env_p and os.path.isfile(env_p):
        return "env"
    reg = load_registry()
    fn = reg.get(role)
    if fn:
        safe = os.path.basename(str(fn))
        fp = os.path.join(get_models_dir(), safe)
        if os.path.isfile(fp):
            return "registry"
    return "missing"


def _providers() -> List[str]:
    available = ort.get_available_providers()
    preferred: List[str] = []
    if "CUDAExecutionProvider" in available:
        preferred.append("CUDAExecutionProvider")
    if "CoreMLExecutionProvider" in available:
        preferred.append("CoreMLExecutionProvider")
    preferred.append("CPUExecutionProvider")
    return preferred


def load_metadata(json_path: str) -> Dict[str, Any]:
    if json_path and os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "trait_names" not in meta:
            meta["trait_names"] = TRAIT_NAMES_DEFAULT
        if "input_size" not in meta:
            meta["input_size"] = IMG_SIZE_DEFAULT
        if "grid_size" not in meta:
            meta["grid_size"] = [16, 16]
        return meta
    return {
        "trait_names": list(TRAIT_NAMES_DEFAULT),
        "input_size": IMG_SIZE_DEFAULT,
        "grid_size": [16, 16],
    }


def letterbox(img_bgr: np.ndarray, new_size: int = YOLO_INPUT_SIZE):
    h, w = img_bgr.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ph, pw = new_size - nh, new_size - nw
    top, left = ph // 2, pw // 2
    padded = cv2.copyMakeBorder(
        resized, top, ph - top, left, pw - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, scale, top, left


def _nms(boxes_xywh, scores, iou_thres=YOLO_IOU_THRES):
    if len(boxes_xywh) == 0:
        return []
    idx = cv2.dnn.NMSBoxes(
        [b.tolist() for b in boxes_xywh],
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_thres,
    )
    if isinstance(idx, np.ndarray):
        return idx.flatten().tolist()
    if isinstance(idx, (list, tuple)) and len(idx):
        return [i[0] if isinstance(i, (list, tuple)) else i for i in idx]
    return []


_yolo_format_logged = False


def postprocess_yolo(yolo_out, frame_h, frame_w, scale, pad_top, pad_left, yolo_bbox_format=None):
    pred = yolo_out[0]
    if pred.ndim == 3:
        pred = pred[0]
    need_transpose = pred.shape[0] < pred.shape[1]
    if need_transpose:
        pred = pred.T
    use_format = yolo_bbox_format if yolo_bbox_format is not None else ("xywh" if need_transpose else "xyxy")
    global _yolo_format_logged
    if not _yolo_format_logged:
        log.info("[Perspicuus/YOLO] formato saída: %s", use_format)
        _yolo_format_logged = True

    pred = pred.astype(np.float32)
    cols = pred.shape[1]
    if cols < 6:
        return []

    if use_format == "xyxy":
        x1_lb, y1_lb = pred[:, 0], pred[:, 1]
        x2_lb, y2_lb = pred[:, 2], pred[:, 3]
        conf = pred[:, 4]
        cls_id = pred[:, 5].astype(np.int32)
        keep = (conf >= YOLO_CONF_THRES) & ((cls_id == 0) | (cls_id == 1))
        if not np.any(keep):
            return []
        x1_lb, y1_lb = x1_lb[keep], y1_lb[keep]
        x2_lb, y2_lb = x2_lb[keep], y2_lb[keep]
        conf_k, cls_id_k = conf[keep], cls_id[keep]
        x1 = np.clip((x1_lb - pad_left) / scale, 0, frame_w)
        y1 = np.clip((y1_lb - pad_top) / scale, 0, frame_h)
        x2 = np.clip((x2_lb - pad_left) / scale, 0, frame_w)
        y2 = np.clip((y2_lb - pad_top) / scale, 0, frame_h)
    else:
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        has_obj = cols > 6
        if has_obj:
            obj_conf = pred[:, 4]
            cls_logits = pred[:, 5:]
            cls_id = np.argmax(cls_logits, axis=1)
            cls_conf = cls_logits[np.arange(len(cls_logits)), cls_id]
            conf = obj_conf * cls_conf
        else:
            cls_logits = pred[:, 4:]
            cls_id = np.argmax(cls_logits, axis=1)
            conf = cls_logits[np.arange(len(cls_logits)), cls_id]
        keep = (conf >= YOLO_CONF_THRES) & ((cls_id == 0) | (cls_id == 1))
        if not np.any(keep):
            return []
        cx_k, cy_k = cx[keep], cy[keep]
        bw_k, bh_k = bw[keep], bh[keep]
        conf_k = conf[keep]
        cls_id_k = cls_id[keep]
        x1 = np.clip((cx_k - bw_k / 2 - pad_left) / scale, 0, frame_w)
        y1 = np.clip((cy_k - bh_k / 2 - pad_top) / scale, 0, frame_h)
        x2 = np.clip((cx_k + bw_k / 2 - pad_left) / scale, 0, frame_w)
        y2 = np.clip((cy_k + bh_k / 2 - pad_top) / scale, 0, frame_h)

    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    kept = _nms(boxes_xywh, conf_k)
    return [
        (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]), float(conf_k[i]), int(cls_id_k[i]))
        for i in kept
    ]


def shrink_bbox(x1, y1, x2, y2, factor, frame_w, frame_h, min_side=32):
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    nw = max(min_side, w * factor)
    nh = max(min_side, h * factor)
    x1_new = cx - nw * 0.5
    y1_new = cy - nh * 0.5
    x2_new = cx + nw * 0.5
    y2_new = cy + nh * 0.5
    x1_new = max(0, min(x1_new, frame_w - 1))
    y1_new = max(0, min(y1_new, frame_h - 1))
    x2_new = max(0, min(x2_new, frame_w))
    y2_new = max(0, min(y2_new, frame_h))
    if x2_new <= x1_new:
        x2_new = x1_new + min_side
    if y2_new <= y1_new:
        y2_new = y1_new + min_side
    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)


def crop_padded(frame, x1, y1, x2, y2, pad=CROP_PAD):
    h, w = frame.shape[:2]
    pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
    cx1, cy1 = max(0, x1 - pw), max(0, y1 - ph)
    cx2, cy2 = min(w, x2 + pw), min(h, y2 + ph)
    return frame[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)


class _HeadState:
    __slots__ = ("sess", "trait_names", "img_size", "persp_dtype", "persp_in", "persp_out_scores", "num_traits")

    def __init__(self):
        self.sess = None
        self.trait_names: List[str] = []
        self.img_size = IMG_SIZE_DEFAULT
        self.persp_dtype = np.float32
        self.persp_in = "image"
        self.persp_out_scores = "scores_all"
        self.num_traits = 1


class PerspicuusInferenceEngine:
    """YOLO compartilhado + um ONNX iudicium por vista (lateral / posterior)."""

    VIEWS = ("lateral", "posterior")

    def __init__(self):
        self._yolo_sess = None
        self._yolo_in_name = "images"
        self._yolo_fp16 = False
        self._heads: Dict[str, _HeadState] = {v: _HeadState() for v in self.VIEWS}

    def is_ready(self) -> bool:
        yolo = resolve_model_path("yolo")
        lat = resolve_model_path("lateral")
        post = resolve_model_path("posterior")
        return bool(yolo and ((lat) or (post)))

    def onnx_path_for(self, view: str) -> Optional[str]:
        role = "lateral" if view == "lateral" else "posterior"
        return resolve_model_path(role)

    def _load_yolo(self) -> None:
        if self._yolo_sess is not None:
            return
        path = resolve_model_path("yolo")
        if not path:
            raise FileNotFoundError("YOLO ONNX não encontrado (env ou upload em ml_models)")
        self._yolo_sess = ort.InferenceSession(path, providers=_providers())
        inp0 = self._yolo_sess.get_inputs()[0]
        self._yolo_in_name = inp0.name
        self._yolo_fp16 = "float16" in inp0.type
        log.info("[Perspicuus] YOLO carregado: %s", os.path.basename(path))

    def _load_head(self, view: str) -> _HeadState:
        if view not in self.VIEWS:
            raise ValueError(view)
        st = self._heads[view]
        if st.sess is not None:
            return st
        role_onnx = "lateral" if view == "lateral" else "posterior"
        role_meta = "lateral_meta" if view == "lateral" else "posterior_meta"
        onnx_path = resolve_model_path(role_onnx)
        if not onnx_path:
            raise FileNotFoundError(
                f"ONNX {view} não encontrado (env {ROLE_TO_ENV[role_onnx]} ou upload)"
            )
        meta_path = resolve_model_path(role_meta) or ""
        meta = load_metadata(meta_path)
        st.trait_names = list(meta["trait_names"])
        st.img_size = int(meta["input_size"])
        st.sess = ort.InferenceSession(onnx_path, providers=_providers())
        inp0 = st.sess.get_inputs()[0]
        st.persp_in = inp0.name
        st.persp_dtype = np.float16 if "float16" in inp0.type else np.float32
        outs = [o.name for o in st.sess.get_outputs()]
        st.persp_out_scores = "scores_all" if "scores_all" in outs else outs[0]
        od = st.sess.get_outputs()[0]
        last = od.shape[-1] if len(od.shape) >= 2 else 1
        st.num_traits = int(last) if isinstance(last, (int, np.integer)) else len(st.trait_names)
        if st.num_traits != len(st.trait_names):
            log.warning(
                "[Perspicuus] %s: modelo devolve %s traits; metadata tem %s nomes",
                view, st.num_traits, len(st.trait_names),
            )
        log.info("[Perspicuus] %s iudicium: %s traits", view, st.num_traits)
        return st

    def preprocess_yolo(self, frame_bgr: np.ndarray):
        img, scale, pad_top, pad_left = letterbox(frame_bgr, YOLO_INPUT_SIZE)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        arr = rgb.transpose(2, 0, 1)[np.newaxis]
        if self._yolo_fp16:
            arr = arr.astype(np.float16)
        return arr, scale, pad_top, pad_left

    def preprocess_perspicuus(self, crop_bgr: np.ndarray, img_size: int, persp_dtype) -> np.ndarray:
        if crop_bgr.size == 0:
            raise ValueError("crop vazio")
        pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        pil = pil.resize((img_size, img_size), Image.LANCZOS)
        arr = np.array(pil, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        arr = arr.transpose(2, 0, 1)[np.newaxis]
        arr = arr.astype(persp_dtype)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    def infer_perspicuus(self, st: _HeadState, img_arr: np.ndarray) -> np.ndarray:
        outs = st.sess.run([st.persp_out_scores], {st.persp_in: img_arr})
        scores_all = outs[0]
        scores = np.asarray(scores_all).reshape(-1).astype(np.float32)
        n = st.num_traits
        if len(scores) < n:
            scores = np.pad(scores, (0, n - len(scores)), constant_values=0.0)
        else:
            scores = scores[:n]
        return scores

    def infer_bgr(self, frame_bgr: np.ndarray, view: str) -> Dict[str, Any]:
        """YOLO crop + iudicium; view em lateral|posterior."""
        self._load_yolo()
        st = self._load_head(view)
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        inp_arr, scale, pad_top, pad_left = self.preprocess_yolo(frame_bgr)
        yolo_out = self._yolo_sess.run(None, {self._yolo_in_name: inp_arr})
        yolo_fmt = os.environ.get("PERSPICUUS_YOLO_BBOX_FORMAT", "").strip() or None
        dets = postprocess_yolo(yolo_out, h, w, scale, pad_top, pad_left, yolo_fmt)
        dets = sorted(dets, key=lambda d: d[4], reverse=True)[:YOLO_MAX_DETECTIONS]

        if not dets:
            crop_bgr = cv2.resize(frame_bgr, (st.img_size, st.img_size), interpolation=cv2.INTER_LINEAR)
            yconf = 0.0
            bbox = None
        else:
            best = dets[0]
            x1, y1, x2, y2, conf, _cls = best
            x1, y1, x2, y2 = shrink_bbox(x1, y1, x2, y2, BBOX_SHRINK, w, h)
            crop_bgr, _ = crop_padded(frame_bgr, x1, y1, x2, y2)
            if crop_bgr.size == 0:
                crop_bgr = cv2.resize(frame_bgr, (st.img_size, st.img_size), interpolation=cv2.INTER_LINEAR)
            yconf = float(conf)
            bbox = [int(x1), int(y1), int(x2), int(y2)]

        img_arr = self.preprocess_perspicuus(crop_bgr, st.img_size, st.persp_dtype)
        raw = self.infer_perspicuus(st, img_arr)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        names = st.trait_names[: len(raw)]
        if len(names) < len(raw):
            names.extend(f"T{i+1}" for i in range(len(names), len(raw)))
        traits = {names[i]: float(raw[i]) for i in range(len(raw))}

        return {
            "traits": traits,
            "trait_names": names[: len(raw)],
            "scores_raw": [float(x) for x in raw],
            "yolo_conf": yconf,
            "bbox": bbox,
            "infer_ms": round(infer_ms, 2),
            "error": None,
        }


_ENGINE: Optional[PerspicuusInferenceEngine] = None


def get_engine() -> PerspicuusInferenceEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = PerspicuusInferenceEngine()
    return _ENGINE


def reset_engine() -> None:
    """Após alterar modelos no disco/registry, força recarregar sessões ONNX."""
    global _ENGINE
    _ENGINE = None


def traits_mean_from_frames(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Média por nome de trait em todos os frames com inferência bem-sucedida
    (várias imagens da mesma pose no mesmo evento).
    """
    acc: Dict[str, List[float]] = {}
    for row in rows:
        if row.get("error") or not row.get("traits"):
            continue
        traits = row["traits"]
        if not isinstance(traits, dict):
            continue
        for k, v in traits.items():
            try:
                acc.setdefault(str(k), []).append(float(v))
            except (TypeError, ValueError):
                continue
    return {k: sum(vals) / len(vals) for k, vals in acc.items() if vals}


def resolve_media_path(web_path: str, uploads_root: str) -> Optional[str]:
    """Converte path público /api/perspicuus/media/... em caminho absoluto no disco."""
    if not web_path or not web_path.startswith("/api/perspicuus/media/"):
        return None
    rest = web_path[len("/api/perspicuus/media/") :].lstrip("/")
    if "/" not in rest:
        return None
    a, b = rest.split("/", 1)
    ef = secure_filename(a)
    fn = secure_filename(b)
    fp = os.path.abspath(os.path.join(uploads_root, "perspicuus", ef, fn))
    base = os.path.abspath(os.path.join(uploads_root, "perspicuus", ef))
    if not fp.startswith(base + os.sep) or not os.path.isfile(fp):
        return None
    return fp


def run_inference_for_event(event: Dict[str, Any], uploads_root: str) -> Dict[str, Any]:
    """
    event: dict com chaves lateral_json, posterior_json (strings JSON), event_id, etc.
    Retorna estrutura para gravar em inference_json.
    """
    eng = get_engine()
    out: Dict[str, Any] = {
        "schema_version": 2,
        "event_id": event.get("event_id"),
        "views": {},
        "skipped": {},
        "config_ok": eng.is_ready(),
    }
    if not eng.is_ready():
        out["error"] = (
            "Modelos não configurados (env PERSPICUUS_* ou upload em Perspicuus → Modelos MK1)"
        )
        out["inferred_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return out

    for view in ("lateral", "posterior"):
        key = f"{view}_json"
        raw = event.get(key) or "[]"
        try:
            frames = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            frames = []
        if not isinstance(frames, list):
            frames = []
        if not eng.onnx_path_for(view):
            role = "lateral" if view == "lateral" else "posterior"
            out["skipped"][view] = (
                f"ONNX {view} em falta (env {ROLE_TO_ENV[role]} ou upload)"
            )
            continue

        view_rows: List[Dict[str, Any]] = []

        for fr in frames:
            if not isinstance(fr, dict):
                continue
            idx = fr.get("frame_index", 0)
            p = fr.get("path", "")
            row: Dict[str, Any] = {
                "frame_index": idx,
                "path": p,
                "traits": {},
                "trait_names": [],
                "yolo_conf": None,
                "bbox": None,
                "infer_ms": None,
                "error": None,
            }
            disk = resolve_media_path(str(p), uploads_root) if p else None
            if not disk:
                row["error"] = "arquivo_nao_encontrado_ou_path_invalido"
                view_rows.append(row)
                continue
            img = cv2.imread(disk)
            if img is None:
                row["error"] = "cv2_imread_falhou"
                view_rows.append(row)
                continue
            try:
                r = eng.infer_bgr(img, view)
                row["traits"] = r["traits"]
                row["trait_names"] = r["trait_names"]
                row["yolo_conf"] = r["yolo_conf"]
                row["bbox"] = r["bbox"]
                row["infer_ms"] = r["infer_ms"]
            except Exception as ex:  # noqa: BLE001
                log.exception("Inferência %s falhou", view)
                row["error"] = str(ex)
            view_rows.append(row)

        traits_mean = traits_mean_from_frames(view_rows)
        n_ok = sum(
            1
            for r in view_rows
            if not r.get("error") and r.get("traits")
        )
        out["views"][view] = {
            "frames": view_rows,
            "traits_mean": traits_mean,
            "n_frames_inferred": n_ok,
        }

    out["inferred_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return out
