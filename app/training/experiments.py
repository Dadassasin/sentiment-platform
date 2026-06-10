from __future__ import annotations

from collections import Counter

from app.models.sentiment import AnalysisResult, MODEL_PROFILES


def summarize_results(results: list[AnalysisResult]) -> dict[str, float | int]:
    total = len(results)
    if total == 0:
        return {"total": 0, "processed": 0, "avg_confidence": 0.0}

    return {
        "total": total,
        "processed": total,
        "avg_confidence": round(sum(result.confidence for result in results) / total, 3),
    }


def class_distribution(results: list[AnalysisResult]) -> Counter[str]:
    return Counter(result.sentiment for result in results)


def comparison_rows() -> list[dict[str, float | int | str]]:
    return MODEL_PROFILES.copy()

