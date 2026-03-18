from __future__ import annotations


def combine_scores(
    nlp_score: float | None,
    vision_score: float | None,
    weight_nlp: float = 0.5,
    weight_vision: float = 0.5,
) -> float | None:
    if nlp_score is None and vision_score is None:
        return None
    if nlp_score is None:
        return float(vision_score)
    if vision_score is None:
        return float(nlp_score)

    total = weight_nlp + weight_vision
    if total <= 0:
        total = 1.0
    return float((nlp_score * weight_nlp + vision_score * weight_vision) / total)
