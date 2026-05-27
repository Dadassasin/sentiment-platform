"""Simple concept drift signals for predicted probabilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.models.sentiment import AnalysisResult


@dataclass(slots=True)
class DriftPoint:
    batch: int
    avg_confidence: float
    class_shares: dict[str, float]


@dataclass(slots=True)
class DriftReport:
    points: list[DriftPoint]
    warning: bool
    message: str


def build_drift_report(results: list[AnalysisResult], batch_size: int = 100) -> DriftReport:
    if not results:
        return DriftReport([], False, "Недостаточно данных для мониторинга.")

    points: list[DriftPoint] = []
    for offset in range(0, len(results), batch_size):
        batch = results[offset : offset + batch_size]
        total = len(batch)
        counts = Counter(item.sentiment for item in batch)
        points.append(
            DriftPoint(
                batch=(offset // batch_size) + 1,
                avg_confidence=round(sum(item.confidence for item in batch) / total, 3),
                class_shares={label: round(count / total, 3) for label, count in counts.items()},
            )
        )

    warning = len(points) >= 2 and abs(points[-1].avg_confidence - points[0].avg_confidence) >= 0.18
    message = (
        "Обнаружен риск дрейфа: средняя уверенность заметно изменилась между батчами."
        if warning
        else "Критичных признаков дрейфа по текущей выборке нет."
    )
    return DriftReport(points, warning, message)
