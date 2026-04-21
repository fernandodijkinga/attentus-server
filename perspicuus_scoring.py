"""
Escala de apresentação dos traits Perspicuus MK1 (sem dependências de inferência).

Raw do modelo na gama nominal −4…+4; apresentação adicional em 1…9 com passo 0,5.
"""

from __future__ import annotations

from typing import Any, Dict

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


def rescale_perspicuus_trait_score(raw: float) -> float:
    """
    Mapeamento linear de [raw_min, raw_max] para [res_min, res_max],
    depois arredondamento ao múltiplo de 0,5 mais próximo e clamp na gama 1…9.
    """
    lo, hi = PERSPICUUS_RAW_SCORE_MIN, PERSPICUUS_RAW_SCORE_MAX
    out_lo, out_hi = PERSPICUUS_RES_SCORE_MIN, PERSPICUUS_RES_SCORE_MAX
    step = PERSPICUUS_RES_SCORE_STEP
    x = max(lo, min(hi, float(raw)))
    if hi <= lo:
        return out_lo
    span_in = hi - lo
    span_out = out_hi - out_lo
    continuous = out_lo + (x - lo) / span_in * span_out
    rounded = round(continuous / step) * step
    return float(max(out_lo, min(out_hi, rounded)))


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


def traits_mean_rescaled_from_mean(traits_mean: Dict[str, float]) -> Dict[str, float]:
    if not isinstance(traits_mean, dict):
        return {}
    return {str(k): rescale_perspicuus_trait_score(float(v)) for k, v in traits_mean.items()}
