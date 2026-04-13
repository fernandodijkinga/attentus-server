"""
THI (Temperature Humidity Index) para bovinos leiteiros — fórmula clássica Holstein.
Faixas de estresse térmico ambiental (gráfico de referência) e limiares por categoria animal.

Módulo de raiz (ficheiro único) para deploy em Render sem pacote `thi/` em falta no Git.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# ─── Classificação por categoria (normal / alerta / estresse) ─────────────────


@dataclass(frozen=True)
class THICategoryThreshold:
    normal_max: float
    alert_max: float


THI_THRESHOLDS: Dict[str, THICategoryThreshold] = {
    "calf": THICategoryThreshold(normal_max=65.0, alert_max=69.0),
    "heifer": THICategoryThreshold(normal_max=70.0, alert_max=73.0),
    "low_lactation": THICategoryThreshold(normal_max=68.0, alert_max=71.0),
    "high_lactation": THICategoryThreshold(normal_max=65.0, alert_max=67.0),
    "primiparous_low": THICategoryThreshold(normal_max=68.0, alert_max=71.0),
    "primiparous_high": THICategoryThreshold(normal_max=65.0, alert_max=67.0),
    "multiparous_low": THICategoryThreshold(normal_max=68.0, alert_max=71.0),
    "multiparous_high": THICategoryThreshold(normal_max=65.0, alert_max=67.0),
}


def calculate_thi(temp_c: float, rh_percent: float) -> float:
    """THI clássico (°C, UR %)."""
    return (1.8 * temp_c + 32) - (
        (0.55 - 0.0055 * rh_percent) * (1.8 * temp_c - 26.0)
    )


def classify_thi(thi: float, category: str) -> str:
    if category not in THI_THRESHOLDS:
        valid = ", ".join(sorted(THI_THRESHOLDS.keys()))
        raise ValueError(f"Categoria inválida: {category}. Válidas: {valid}")
    t = THI_THRESHOLDS[category]
    if thi < t.normal_max:
        return "normal"
    if thi <= t.alert_max:
        return "alert"
    return "stress"


def evaluate_animal(temp_c: float, rh_percent: float, category: str) -> Dict[str, Any]:
    thi = calculate_thi(temp_c, rh_percent)
    status = classify_thi(thi, category)
    return {
        "temperature_c": round(temp_c, 2),
        "humidity_percent": round(rh_percent, 2),
        "category": category,
        "thi": round(thi, 2),
        "status": status,
    }


def _hex_to_rgba(hex_color: str, alpha: float = 0.38) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


_THI_ZONE_DEF: List[Tuple[float, float, str, str]] = [
    (0.0, 68.0, "Sem estresse térmico (<68)", "#E2EFDA"),
    (68.0, 72.0, "Estresse leve (68–71)", "#FFF2CC"),
    (72.0, 80.0, "Estresse moderado (72–79)", "#FFC000"),
    (80.0, 90.0, "Estresse forte (80–89)", "#ED7D31"),
    (90.0, 100.0, "Estresse severo (90–99)", "#FF0000"),
    (100.0, 130.0, "Estresse fatal (>100)", "#800000"),
]


def thi_holstein_zones_for_chart() -> List[Dict[str, Any]]:
    """Metadados para fundo do gráfico THI (y_min/y_max em índice THI)."""
    out: List[Dict[str, Any]] = []
    for y0, y1, label, hx in _THI_ZONE_DEF:
        out.append(
            {
                "y_min": y0,
                "y_max": y1,
                "label": label,
                "hex": hx,
                "rgba": _hex_to_rgba(hx, 0.38),
            }
        )
    return out


def thi_thresholds_json() -> Dict[str, Dict[str, float]]:
    return {k: {"normal_max": v.normal_max, "alert_max": v.alert_max} for k, v in THI_THRESHOLDS.items()}
