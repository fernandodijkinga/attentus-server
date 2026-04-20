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
    Faixa esperada do modelo bruto: -4 .. +4
      -4 -> 1.00
       0 -> 3.00
      +4 -> 5.00
    """
    if raw_score is None:
        return None
    x = float(raw_score)
    # Normaliza -4..+4 para 1..5 de forma linear: 1 + (x + 4) / 2
    x = 1.0 + (max(-4.0, min(4.0, x)) + 4.0) / 2.0
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


def save_ecc_bbox_overlay(
    abs_path: str,
    bbox: Any,
    boxed_abs_path: str,
    yolo_conf: Any = None,
) -> bool:
    """
    Salva uma cópia da imagem original com bounding box desenhado.
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

    out = img.copy()
    color = (0, 229, 160)  # BGR (verde accent)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    label = "YOLO crop"
    if yolo_conf is not None:
        try:
            label += f" {float(yolo_conf):.2f}"
        except (TypeError, ValueError):
            pass
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_top = max(0, y1 - th - base - 6)
    x_right = min(w - 1, x1 + tw + 8)
    cv2.rectangle(out, (x1, y_top), (x_right, y1), color, -1)
    cv2.putText(
        out, label, (x1 + 4, max(10, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (5, 12, 20), 1, cv2.LINE_AA
    )
    return bool(cv2.imwrite(boxed_abs_path, out))


def ecc_farm_time_series(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Agrega por (fazenda, data de inferência).

    - n_records: todos os registros naquele dia (uploads), mesmo sem ecc_score.
    - n_animals: brincos distintos não vazios.
    - mean / std: apenas a partir de registros com ecc_score numérico; se não houver,
      vêm como None (útil para manter o dia na série de volume).
    """
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        farm = str(r.get("farm_id") or "")
        d = str(r.get("inference_date") or "")
        if not farm or not d:
            continue
        key = (farm, d)
        cell = by_key.setdefault(key, {"vals": [], "animals": set(), "n_records": 0})
        cell["n_records"] = int(cell["n_records"]) + 1
        tag = str(r.get("animal_tag") or "").strip()
        if tag:
            cell["animals"].add(tag)
        if r.get("ecc_score") is not None:
            try:
                cell["vals"].append(float(r["ecc_score"]))
            except (TypeError, ValueError):
                pass

    out: list[dict[str, Any]] = []
    for (farm, d), cell in by_key.items():
        vals = cell["vals"]
        n_animals = len([a for a in cell["animals"] if a])
        n_records = int(cell["n_records"])
        if vals:
            n = len(vals)
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            mean_r = round(mean, 3)
            std_r = round(math.sqrt(var), 3)
        else:
            mean_r, std_r = None, None
        out.append(
            {
                "farm_id": farm,
                "date": d,
                "mean": mean_r,
                "std": std_r,
                "n_records": n_records,
                "n_animals": n_animals,
                "n_scored": len(vals),
            }
        )

    out.sort(key=lambda x: (x["farm_id"], x["date"]))
    return out


def ecc_attention_ranking(
    rows: list[dict[str, Any]],
    farm_id: str,
    *,
    min_records: int = 2,
    top_n: int = 50,
    sort_by: str = "spread",
) -> list[dict[str, Any]]:
    """
    Animais de uma fazenda com maior variação de ECC ao longo do tempo.

    - spread: max(ecc) - min(ecc) (amplitude)
    - max_step: maior |ΔECC| entre duas medições consecutivas (por data/ordem)
    """
    farm_id = str(farm_id or "").strip()
    if not farm_id:
        return []

    by_tag: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if str(r.get("farm_id") or "") != farm_id:
            continue
        if r.get("ecc_score") is None:
            continue
        try:
            sc = float(r["ecc_score"])
        except (TypeError, ValueError):
            continue
        tag = str(r.get("animal_tag") or "").strip()
        if not tag:
            continue
        by_tag.setdefault(tag, []).append({**r, "_ecc": sc})

    out: list[dict[str, Any]] = []
    for tag, recs in by_tag.items():
        recs.sort(
            key=lambda x: (str(x.get("inference_date") or ""), int(x.get("id") or 0))
        )
        if len(recs) < min_records:
            continue
        scores = [float(x["_ecc"]) for x in recs]
        lo, hi = min(scores), max(scores)
        spread = hi - lo
        max_step = 0.0
        for i in range(1, len(scores)):
            max_step = max(max_step, abs(scores[i] - scores[i - 1]))
        if spread < 1e-9 and max_step < 1e-9:
            continue
        d0 = str(recs[0].get("inference_date") or "")
        d1 = str(recs[-1].get("inference_date") or "")
        out.append(
            {
                "animal_tag": tag,
                "n_records": len(recs),
                "min_ecc": round(lo, 2),
                "max_ecc": round(hi, 2),
                "spread": round(spread, 2),
                "max_step": round(max_step, 2),
                "first_date": d0,
                "last_date": d1,
            }
        )

    key = "max_step" if sort_by == "step" else "spread"
    out.sort(key=lambda x: -x[key])
    return out[:top_n]

