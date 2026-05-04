"""
Escala de apresentação dos **traits** Perspicuus MK1 (sem dependências de inferência).

Raw do modelo na gama nominal −4…+4; apresentação em 1…9 com passo 0,5.
- **Holstein / JSON de evento**: mapeamento fixo −4…+4 → 1…9 (`traits_rescaled_from_traits`).
- **Angus / Nelore**: a app pode usar **min/max por trait no lote** (com shrink para −4…+4),
  mesma família de parâmetros que o escore composto (`PERSPICUUS_LOT_MIN_N_PURE`, `PERSPICUUS_LOT_BLEND_K`).

O escore composto `raw_score` na BD continua com política própria em `app._perspicuus_rescale_with_cal`.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

PERSPICUUS_RAW_SCORE_MIN = -4.0
PERSPICUUS_RAW_SCORE_MAX = 4.0
PERSPICUUS_RES_SCORE_MIN = 1.0
PERSPICUUS_RES_SCORE_MAX = 9.0
PERSPICUUS_RES_SCORE_STEP = 0.5
TRAIT_SCORE_SCALE_META: Dict[str, Any] = {
    "raw_min": PERSPICUUS_RAW_SCORE_MIN,
    "raw_max": PERSPICUUS_RAW_SCORE_MAX,
    "rescaled_min": PERSPICUUS_RES_SCORE_MIN,
    "rescaled_max": PERSPICUUS_RES_SCORE_MAX,
    "rescaled_step": PERSPICUUS_RES_SCORE_STEP,
}


def rescale_perspicuus_trait_score_with_bounds(
    raw: float,
    raw_min: float,
    raw_max: float,
    *,
    out_lo: float = PERSPICUUS_RES_SCORE_MIN,
    out_hi: float = PERSPICUUS_RES_SCORE_MAX,
    step: float = PERSPICUUS_RES_SCORE_STEP,
) -> float:
    """Linear [raw_min, raw_max] → [out_lo, out_hi], passo 0,5 e clamp 1…9."""
    lo, hi = float(raw_min), float(raw_max)
    ol, oh = float(out_lo), float(out_hi)
    st = float(step)
    if hi <= lo:
        return ol
    if oh <= ol:
        return ol
    if st <= 0:
        st = PERSPICUUS_RES_SCORE_STEP
    x = max(lo, min(hi, float(raw)))
    span_in = hi - lo
    span_out = oh - ol
    continuous = ol + (x - lo) / span_in * span_out
    rounded = round(continuous / st) * st
    return float(max(ol, min(oh, rounded)))


def rescale_perspicuus_trait_score(raw: float) -> float:
    """Mapeamento nominal −4…+4 → 1…9 (passo 0,5)."""
    return rescale_perspicuus_trait_score_with_bounds(
        raw, PERSPICUUS_RAW_SCORE_MIN, PERSPICUUS_RAW_SCORE_MAX
    )


def traits_rescaled_from_traits(traits: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(traits, dict):
        return out
    for k, v in traits.items():
        try:
            out[str(k)] = rescale_perspicuus_trait_score(float(v))
        except (TypeError, ValueError):
            continue
    return out


def traits_rescaled_with_per_trait_bounds(
    traits: Dict[str, Any],
    bounds: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    """
    Reescala cada trait com o seu par (raw_min, raw_max); chaves em falta usam −4…+4.
    ``bounds`` vem tipicamente de min/max observados no lote (após shrink) em Angus/Nelore.
    """
    out: Dict[str, float] = {}
    if not isinstance(traits, dict):
        return out
    default: Tuple[float, float] = (PERSPICUUS_RAW_SCORE_MIN, PERSPICUUS_RAW_SCORE_MAX)
    for k, v in traits.items():
        ks = str(k)
        try:
            lo, hi = bounds.get(ks, default)
            lo, hi = float(lo), float(hi)
            if hi <= lo:
                lo, hi = default
            out[ks] = rescale_perspicuus_trait_score_with_bounds(float(v), lo, hi)
        except (TypeError, ValueError):
            continue
    return out


def traits_mean_rescaled_from_mean(traits_mean: Dict[str, float]) -> Dict[str, float]:
    if not isinstance(traits_mean, dict):
        return {}
    return {str(k): rescale_perspicuus_trait_score(float(v)) for k, v in traits_mean.items()}
