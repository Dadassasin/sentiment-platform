"""Simple monitoring summary for analysis results."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.models.sentiment import AnalysisResult, normalize_comparable_label


UNCERTAIN_RATE_WARNING = 0.30
CONFIDENCE_WARNING = 0.65
DOMINANT_CLASS_WARNING = 0.80
MIN_MONITORING_SAMPLE = 20


@dataclass(slots=True, frozen=True)
class MonitoringSummary:
    total: int
    positive_count: int
    neutral_count: int
    negative_count: int
    uncertain_count: int
    avg_confidence: float
    class_counts: Counter[str]
    other_counts: Counter[str]

    @property
    def uncertain_rate(self) -> float:
        return self.uncertain_count / self.total if self.total else 0.0

    @property
    def positive_share(self) -> float:
        return self.positive_count / self.total if self.total else 0.0

    @property
    def neutral_share(self) -> float:
        return self.neutral_count / self.total if self.total else 0.0

    @property
    def negative_share(self) -> float:
        return self.negative_count / self.total if self.total else 0.0

    @property
    def dominant_class(self) -> tuple[str, int, float]:
        if not self.class_counts or self.total == 0:
            return "-", 0, 0.0
        label, count = self.class_counts.most_common(1)[0]
        return label, count, count / self.total

    @property
    def status(self) -> str:
        if self.total == 0:
            return "Нет данных"
        if self.uncertain_rate >= UNCERTAIN_RATE_WARNING:
            return "Требуется ручная проверка: много неуверенных ответов"
        if self.dominant_class[2] >= DOMINANT_CLASS_WARNING and self.total >= MIN_MONITORING_SAMPLE:
            return "Проверьте выборку: сильный перекос в один класс"
        if self.avg_confidence < CONFIDENCE_WARNING:
            return "Проверьте модель: низкая средняя уверенность"
        return "Стабильно"

    @property
    def risk_index(self) -> int:
        if self.total == 0:
            return 0

        score = 0
        if self.uncertain_rate >= UNCERTAIN_RATE_WARNING:
            score += 35
        elif self.uncertain_rate >= 0.15:
            score += 20

        if self.avg_confidence < CONFIDENCE_WARNING:
            score += 30
        elif self.avg_confidence < 0.75:
            score += 15

        dominant_share = self.dominant_class[2]
        if dominant_share >= DOMINANT_CLASS_WARNING and self.total >= MIN_MONITORING_SAMPLE:
            score += 25
        elif dominant_share >= 0.65 and self.total >= MIN_MONITORING_SAMPLE:
            score += 10

        if self.total < MIN_MONITORING_SAMPLE:
            score += 10

        return min(score, 100)

    @property
    def risk_level(self) -> str:
        if self.total == 0:
            return "нет данных"
        if self.risk_index >= 60:
            return "высокий"
        if self.risk_index >= 30:
            return "средний"
        return "низкий"

    @property
    def warnings(self) -> list[str]:
        if self.total == 0:
            return []

        warnings: list[str] = []
        if self.total < MIN_MONITORING_SAMPLE:
            warnings.append("Мало данных для уверенного вывода по мониторингу.")
        if self.uncertain_rate >= UNCERTAIN_RATE_WARNING:
            warnings.append("Высокая доля неуверенных предсказаний.")
        if self.avg_confidence < CONFIDENCE_WARNING:
            warnings.append("Средняя уверенность модели ниже контрольного порога.")
        dominant_label, _, dominant_share = self.dominant_class
        if dominant_share >= DOMINANT_CLASS_WARNING and self.total >= MIN_MONITORING_SAMPLE:
            warnings.append(f"Сильный перекос в класс '{dominant_label}'.")
        return warnings

    @property
    def recommendation(self) -> str:
        if self.total == 0:
            return "Запустите анализ, чтобы сформировать мониторинг."
        if self.uncertain_rate >= UNCERTAIN_RATE_WARNING:
            return "Проверить вручную примеры с низкой уверенностью и добавить их в разметку."
        if self.avg_confidence < CONFIDENCE_WARNING:
            return "Проверить выбранную модель и соответствие входных данных обучающей выборке."
        if self.dominant_class[2] >= DOMINANT_CLASS_WARNING and self.total >= MIN_MONITORING_SAMPLE:
            return "Проверить входной датасет на перекос или однотипные тексты."
        return "Критичных действий не требуется; продолжайте наблюдение на следующих запусках."

    @property
    def message(self) -> str:
        if self.total == 0:
            return "Недостаточно данных для мониторинга."
        return f"Контроль результата: {self.status}."


@dataclass(slots=True, frozen=True)
class MonitoringSnapshot:
    created_at: str
    model_name: str
    total: int
    avg_confidence: float
    uncertain_rate: float
    positive_share: float
    neutral_share: float
    negative_share: float
    dominant_label: str
    dominant_share: float
    risk_index: int


@dataclass(slots=True, frozen=True)
class MonitoringComparison:
    previous: MonitoringSnapshot | None
    confidence_delta: float = 0.0
    uncertain_rate_delta: float = 0.0
    positive_share_delta: float = 0.0
    neutral_share_delta: float = 0.0
    negative_share_delta: float = 0.0
    risk_index_delta: int = 0

    @property
    def available(self) -> bool:
        return self.previous is not None


def build_monitoring_summary(
    results: list[AnalysisResult],
    confidence_threshold: float = 0.6,
) -> MonitoringSummary:
    total = len(results)
    class_counts = Counter(result.sentiment for result in results)
    normalized_counts = Counter(normalize_comparable_label(result.sentiment) for result in results)
    known_labels = {"positive", "neutral", "negative"}
    other_counts = Counter(
        {
            label: count
            for label, count in class_counts.items()
            if normalize_comparable_label(label) not in known_labels
        }
    )

    avg_confidence = sum(result.confidence for result in results) / total if total else 0.0
    uncertain_count = sum(result.confidence < confidence_threshold for result in results)

    return MonitoringSummary(
        total=total,
        positive_count=normalized_counts.get("positive", 0),
        neutral_count=normalized_counts.get("neutral", 0),
        negative_count=normalized_counts.get("negative", 0),
        uncertain_count=uncertain_count,
        avg_confidence=avg_confidence,
        class_counts=class_counts,
        other_counts=other_counts,
    )


def create_monitoring_snapshot(summary: MonitoringSummary, model_name: str) -> MonitoringSnapshot:
    dominant_label, _, dominant_share = summary.dominant_class
    return MonitoringSnapshot(
        created_at=datetime.now().isoformat(timespec="seconds"),
        model_name=model_name,
        total=summary.total,
        avg_confidence=summary.avg_confidence,
        uncertain_rate=summary.uncertain_rate,
        positive_share=summary.positive_share,
        neutral_share=summary.neutral_share,
        negative_share=summary.negative_share,
        dominant_label=dominant_label,
        dominant_share=dominant_share,
        risk_index=summary.risk_index,
    )


def compare_with_snapshot(summary: MonitoringSummary, previous: MonitoringSnapshot | None) -> MonitoringComparison:
    if previous is None:
        return MonitoringComparison(previous=None)

    return MonitoringComparison(
        previous=previous,
        confidence_delta=summary.avg_confidence - previous.avg_confidence,
        uncertain_rate_delta=summary.uncertain_rate - previous.uncertain_rate,
        positive_share_delta=summary.positive_share - previous.positive_share,
        neutral_share_delta=summary.neutral_share - previous.neutral_share,
        negative_share_delta=summary.negative_share - previous.negative_share,
        risk_index_delta=summary.risk_index - previous.risk_index,
    )


def load_monitoring_snapshot(path: str | Path) -> MonitoringSnapshot | None:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        return None
    try:
        data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    try:
        return MonitoringSnapshot(
            created_at=str(data.get("created_at", "")),
            model_name=str(data.get("model_name", "")),
            total=int(data.get("total", 0)),
            avg_confidence=float(data.get("avg_confidence", 0.0)),
            uncertain_rate=float(data.get("uncertain_rate", 0.0)),
            positive_share=float(data.get("positive_share", 0.0)),
            neutral_share=float(data.get("neutral_share", 0.0)),
            negative_share=float(data.get("negative_share", 0.0)),
            dominant_label=str(data.get("dominant_label", "-")),
            dominant_share=float(data.get("dominant_share", 0.0)),
            risk_index=int(data.get("risk_index", 0)),
        )
    except (TypeError, ValueError):
        return None


def save_monitoring_snapshot(path: str | Path, snapshot: MonitoringSnapshot) -> None:
    snapshot_path = Path(path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(
            {
                "created_at": snapshot.created_at,
                "model_name": snapshot.model_name,
                "total": snapshot.total,
                "avg_confidence": snapshot.avg_confidence,
                "uncertain_rate": snapshot.uncertain_rate,
                "positive_share": snapshot.positive_share,
                "neutral_share": snapshot.neutral_share,
                "negative_share": snapshot.negative_share,
                "dominant_label": snapshot.dominant_label,
                "dominant_share": snapshot.dominant_share,
                "risk_index": snapshot.risk_index,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
