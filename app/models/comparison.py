"""Utilities for comparing multiple local models on one dataset."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

from app.models.sentiment import (
    AnalysisResult,
    ModelSchema,
    SentimentAnalyzer,
    inspect_model_schema,
    normalize_comparable_label,
)
from app.preprocessing import PreprocessingOptions

try:
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:  # pragma: no cover - optional dependency guard
    accuracy_score = None
    f1_score = None


@dataclass(slots=True)
class ModelBehaviorResult:
    model_name: str
    schema: ModelSchema
    sample_count: int
    avg_confidence: float
    median_confidence: float
    low_confidence_rate: float
    prediction_entropy: float
    top_class_share: float
    inference_seconds: float
    predictions: list[AnalysisResult]


@dataclass(slots=True)
class ModelQualityResult:
    model_name: str
    schema_signature: str
    label_count: int
    sample_count: int
    accuracy: float
    macro_f1: float


@dataclass(slots=True)
class ComparisonDisagreement:
    text: str
    source: str
    predictions: dict[str, str]
    confidences: dict[str, float]


@dataclass(slots=True)
class ComparisonRunResult:
    behavior_rows: list[ModelBehaviorResult]
    quality_rows: list[ModelQualityResult]
    disagreements: list[ComparisonDisagreement]
    quality_groups: dict[str, list[str]]
    quality_notes: list[str]


def compare_models(
    model_names: list[str],
    texts: list[str],
    options: PreprocessingOptions,
    device: str,
    true_labels: list[str] | None = None,
    sources: list[str] | None = None,
    batch_size: int | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> ComparisonRunResult:
    behavior_rows: list[ModelBehaviorResult] = []
    quality_rows: list[ModelQualityResult] = []
    quality_groups: dict[str, list[str]] = defaultdict(list)
    quality_notes: list[str] = []

    for model_name in model_names:
        schema = inspect_model_schema(model_name)
        analyzer = SentimentAnalyzer(model_name, options, device)
        started_at = time.perf_counter()
        predictions = analyzer.analyze(
            texts,
            sources,
            batch_size=batch_size,
            progress_callback=(
                (lambda processed, total, current_model=model_name: progress_callback(current_model, processed, total))
                if progress_callback is not None
                else None
            ),
        )
        elapsed = time.perf_counter() - started_at
        behavior_rows.append(_build_behavior_result(model_name, schema, predictions, elapsed))
        if schema.labels:
            quality_groups[schema.signature].append(model_name)

    if true_labels:
        if accuracy_score is None or f1_score is None:
            quality_notes.append("Для сравнения качества не установлен scikit-learn.")
        else:
            normalized_true_labels = [normalize_comparable_label(label) for label in true_labels if label]
            for row in behavior_rows:
                schema_labels_raw = list(row.schema.labels)
                if not schema_labels_raw:
                    quality_notes.append(f"{row.model_name}: схема меток не найдена, метрики качества пропущены.")
                    continue
                schema_labels = _unique_preserving_order(normalize_comparable_label(label) for label in schema_labels_raw)
                true_label_set = {label for label in normalized_true_labels if label}
                schema_set = set(schema_labels)
                if not true_label_set or not schema_labels:
                    continue
                if not true_label_set.issubset(schema_set):
                    quality_notes.append(f"{row.model_name}: метки датасета не совпадают с метками модели.")
                    continue
                predicted_labels = [normalize_comparable_label(item.sentiment) for item in row.predictions]
                quality_rows.append(
                    ModelQualityResult(
                        model_name=row.model_name,
                        schema_signature=row.schema.signature,
                        label_count=len(row.schema.labels),
                        sample_count=len(predicted_labels),
                        accuracy=float(accuracy_score(normalized_true_labels, predicted_labels)),
                        macro_f1=float(
                            f1_score(normalized_true_labels, predicted_labels, labels=schema_labels, average="macro")
                        ),
                    )
                )
    else:
        quality_notes.append("Колонка истинной метки не выбрана: доступно только сравнение поведения.")

    disagreements = _build_disagreements(behavior_rows)
    return ComparisonRunResult(
        behavior_rows=behavior_rows,
        quality_rows=quality_rows,
        disagreements=disagreements,
        quality_groups=dict(quality_groups),
        quality_notes=quality_notes,
    )


def _build_behavior_result(
    model_name: str,
    schema: ModelSchema,
    predictions: list[AnalysisResult],
    elapsed: float,
) -> ModelBehaviorResult:
    confidences = sorted(item.confidence for item in predictions)
    counts = Counter(item.sentiment for item in predictions)
    total = len(predictions)
    avg_confidence = sum(confidences) / total if total else 0.0
    low_confidence_rate = sum(value < 0.60 for value in confidences) / total if total else 0.0
    median_confidence = confidences[total // 2] if total else 0.0
    if total and total % 2 == 0:
        median_confidence = (confidences[(total // 2) - 1] + confidences[total // 2]) / 2
    prediction_entropy = _normalized_entropy(counts, total)
    top_class_share = (counts.most_common(1)[0][1] / total) if total else 0.0
    return ModelBehaviorResult(
        model_name=model_name,
        schema=schema,
        sample_count=total,
        avg_confidence=avg_confidence,
        median_confidence=median_confidence,
        low_confidence_rate=low_confidence_rate,
        prediction_entropy=prediction_entropy,
        top_class_share=top_class_share,
        inference_seconds=elapsed,
        predictions=predictions,
    )


def _normalized_entropy(counts: Counter[str], total: int) -> float:
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    max_entropy = math.log2(len(counts))
    return entropy / max_entropy if max_entropy else 0.0


def _unique_preserving_order(values: list[str] | tuple[str, ...] | object) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _build_disagreements(rows: list[ModelBehaviorResult], limit: int = 100) -> list[ComparisonDisagreement]:
    if len(rows) < 2 or not rows[0].predictions:
        return []

    disagreements: list[ComparisonDisagreement] = []
    size = min(len(row.predictions) for row in rows)
    for index in range(size):
        predictions = {row.model_name: row.predictions[index].sentiment for row in rows}
        if len(set(predictions.values())) <= 1:
            continue
        confidences = {row.model_name: row.predictions[index].confidence for row in rows}
        text = rows[0].predictions[index].text
        source = rows[0].predictions[index].source
        disagreements.append(
            ComparisonDisagreement(
                text=text,
                source=source,
                predictions=predictions,
                confidences=confidences,
            )
        )
        if len(disagreements) >= limit:
            break
    return disagreements
