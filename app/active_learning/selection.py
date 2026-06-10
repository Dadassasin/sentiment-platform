"""Selection of low-confidence examples for manual labeling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


ProbabilityRow = Mapping[str, float] | Sequence[float]


@dataclass(slots=True, frozen=True)
class UncertainExample:
    index: int
    text: str
    max_probability: float
    probabilities: ProbabilityRow


def select_uncertain_examples(
    texts: Sequence[object],
    probabilities: Sequence[ProbabilityRow],
    threshold: float = 0.6,
    *,
    verbose: bool = False,
) -> list[UncertainExample]:
    """Return texts whose highest predicted probability is below threshold."""

    uncertain: list[UncertainExample] = []
    for index, (text, probability_row) in enumerate(zip(texts, probabilities, strict=False)):
        max_probability = max_prediction_probability(probability_row)
        if max_probability < threshold:
            uncertain.append(
                UncertainExample(
                    index=index,
                    text="" if text is None else str(text),
                    max_probability=max_probability,
                    probabilities=probability_row,
                )
            )

    if verbose:
        print(f"Найдено неуверенных примеров: {len(uncertain)}")

    return uncertain


def max_prediction_probability(probabilities: ProbabilityRow) -> float:
    if isinstance(probabilities, Mapping):
        values = probabilities.values()
    else:
        values = probabilities

    numeric_values = [float(value) for value in values]
    return max(numeric_values, default=0.0)
