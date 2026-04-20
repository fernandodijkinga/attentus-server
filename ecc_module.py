from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from typing import Any

import cv2
from werkzeug.utils import secure_filename

log = logging.getLogger(__name__)


def safe_slug(v: str, fallback: str = "x") -> str:
    s = secure_filename(str(v or "").strip())
    return s or fallback


def parse_iso_day(v: str) -> str | None:
    s = str(v or "").strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date().isoformat()
    except ValueError:
        return None


def pick_ecc_trait(traits: dict) -> tuple[str | None, float | None]:
    """
    Escolhe trait de ECC/BCS.
    Prioridade: nome contendo ecc/bcs/body_condition.
    """
    if not isinstance(traits, dict) or not traits:
        return None, None
    keys = list(traits.keys())
    preferred = []
    for k in keys:
        lk = str(k).lower()
        if "ecc" in lk or "bcs" in lk or "body" in lk or "condition" in lk:
            preferred.append(k)
    use = preferred[0] if preferred else keys[0]
    try:
        return str(use), float(traits.get(use))
    except (TypeError, ValueError):
        return str(use), None


def rescale_ecc_1_to_5_quarter(raw_score: float | None) -> float | None:
    """
    Reescala para [1, 5] em incrementos de 0.25.
    - Se já estiver em 1..5, mantém.
    - Senão, assume 0..1 e mapeia para 1..5.
    """
    if raw_score is None:
        return None
    x = float(raw_score)
    if x < 1.0 or x > 5.0:
        x = 1.0 + max(0.0, min(1.0, x)) * 4.0
    x = max(1.0, min(5.0, x))
    return round(x * 4.0) / 4.0


def infer_ecc_posterior(abs_path: str) -> dict[str, Any]:
    """
    Pipeline ECC fixo:
      YOLO -> crop -> ONNX posterior -> trait ECC -> escala 1..5 (0.25).
    """
    try:
        from perspicuus_inference import get_engine
    except ImportError as e:
        return {"error": f"Módulo de inferência indisponível: {e}"}

    eng = get_engine()
    if not eng.is_ready():
        return {"error": "Motor Perspicuus não está pronto (YOLO + ONNX)."}
    if not eng.onnx_path_for("posterior"):
        return {"error": "Modelo ONNX posterior não está configurado."}

    img = cv2.imread(abs_path)
    if img is None:
        return {"error": "Falha ao abrir imagem."}

    try:
        out = eng.infer_bgr(img, "posterior")
    except Exception as e:  # noqa: BLE001
        log.exception("ECC inferência posterior falhou em %s", abs_path)
        return {"error": str(e)}

    traits = out.get("traits") or {}
    tname, raw = pick_ecc_trait(traits)
    ecc = rescale_ecc_1_to_5_quarter(raw)
    return {
        "trait_name": tname or "",
        "raw_score": raw,
        "ecc_score": ecc,
        "traits": traits,
        "meta": {
            "yolo_conf": out.get("yolo_conf"),
            "bbox": out.get("bbox"),
            "infer_ms": out.get("infer_ms"),
            "view": "posterior",
        },
        "error": None if ecc is not None else "trait_ecc_ausente_ou_invalida",
    }


def save_ecc_crop_thumbnail(
    abs_path: str,
    bbox: Any,
    thumb_abs_path: str,
    target_size: tuple[int, int] = (224, 224),
) -> bool:
    """
    Salva thumbnail cropada no bbox detectado.
    Retorna False se não conseguir gerar crop válido.
    """
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    try:
        x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
    except (TypeError, ValueError):
        return False

    img = cv2.imread(abs_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return False

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    if target_size and len(target_size) == 2:
        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

    return bool(cv2.imwrite(thumb_abs_path, crop))


def ecc_farm_time_series(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        if r.get("ecc_score") is None:
            continue
        farm = str(r.get("farm_id") or "")
        d = str(r.get("inference_date") or "")
        if not farm or not d:
            continue
        key = (farm, d)
        cell = by_key.setdefault(key, {"vals": [], "animals": set()})
        cell["vals"].append(float(r["ecc_score"]))
        cell["animals"].add(str(r.get("animal_tag") or ""))

    out: list[dict[str, Any]] = []
    for (farm, d), cell in by_key.items():
        vals = cell["vals"]
        if not vals:
            continue
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        out.append(
            {
                "farm_id": farm,
                "date": d,
                "mean": round(mean, 3),
                "std": round(math.sqrt(var), 3),
                "n_records": n,
                "n_animals": len([a for a in cell["animals"] if a]),
            }
        )

    out.sort(key=lambda x: (x["farm_id"], x["date"]))
    return out

