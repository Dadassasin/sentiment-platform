"""Transformer-backed sentiment inference helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.preprocessing import PreprocessingOptions, TextPreprocessor

DEFAULT_HF_HOME = Path.cwd() / "models" / "huggingface"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover - dependency guard
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "flax_model.msgpack",
)

TOKENIZER_VOCAB_FILES = (
    "tokenizer.json",
    "vocab.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tokenizer.model",
)

TRAINING_BASE_MODELS = [
    "cointegrated/rubert-tiny2",
    "DeepPavlov/rubert-base-cased",
    "ai-forever/ruBert-base",
    "xlm-roberta-base",
]

MODEL_PROFILES: list[dict[str, float | int | str]] = []

CANONICAL_TO_RU = {
    "negative": "Отрицательная",
    "neutral": "Нейтральная",
    "positive": "Положительная",
}

RU_TO_CANONICAL = {value.casefold(): key for key, value in CANONICAL_TO_RU.items()}
POSITIVE_KEYS = {"positive", "pos", "label_2", "2", "5"}
NEUTRAL_KEYS = {"neutral", "neu", "label_1", "1", "3", "4"}
NEGATIVE_KEYS = {"negative", "neg", "label_0", "0"}


def normalize_comparable_label(label: object) -> str:
    raw = "" if label is None else str(label).strip()
    if not raw:
        return ""
    key = raw.casefold()
    if key in RU_TO_CANONICAL:
        return RU_TO_CANONICAL[key]
    if key in POSITIVE_KEYS:
        return "positive"
    if key in NEUTRAL_KEYS:
        return "neutral"
    if key in NEGATIVE_KEYS:
        return "negative"
    return raw


class TransformerLoadError(RuntimeError):
    """Raised when transformer assets cannot be loaded or used."""


@dataclass(slots=True)
class AnalysisResult:
    text: str
    sentiment: str
    confidence: float
    probabilities: dict[str, float]
    source: str = ""


@dataclass(slots=True, frozen=True)
class ModelSchema:
    model_name: str
    labels: tuple[str, ...]

    @property
    def signature(self) -> str:
        return " | ".join(self.labels) if self.labels else "неизвестно"


class SentimentAnalyzer:
    """Loads a sequence-classification transformer and predicts sentiments."""

    def __init__(
        self,
        model_name: str,
        options: PreprocessingOptions | None = None,
        device: str = "CPU",
    ) -> None:
        self.model_name = (model_name or "").strip()
        self.options = options or PreprocessingOptions(
            lowercase=False,
            remove_punctuation=False,
            remove_stop_words=False,
            lemmatize=False,
        )
        self.device_name = device
        self.preprocessor = TextPreprocessor(self.options)
        self._tokenizer = None
        self._model = None
        self._torch_device = None

    def analyze(
        self,
        texts: list[object],
        sources: list[object] | None = None,
        batch_size: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[AnalysisResult]:
        if not texts:
            return []

        model, tokenizer, torch_device = self._ensure_model()
        prepared_texts = self.preprocessor.process_many(texts)
        source_values = self._normalize_sources(sources, len(prepared_texts))
        total = len(prepared_texts)
        step = max(1, batch_size or self._default_batch_size())
        results: list[AnalysisResult] = []
        labels: list[str] | None = None

        for start in range(0, total, step):
            end = min(start + step, total)
            encoded = tokenizer(
                prepared_texts[start:end],
                padding=True,
                truncation=True,
                max_length=min(getattr(tokenizer, "model_max_length", 512), 512),
                return_tensors="pt",
            )
            encoded = {key: value.to(torch_device) for key, value in encoded.items()}

            with torch.inference_mode():
                outputs = model(**encoded)
                probabilities_tensor = torch.softmax(outputs.logits, dim=-1).detach().cpu()

            if labels is None:
                labels = self._ordered_labels(model, probabilities_tensor.shape[-1])

            for offset, original_text in enumerate(texts[start:end]):
                row = probabilities_tensor[offset].tolist()
                probability_map = {label: float(score) for label, score in zip(labels, row, strict=False)}
                sentiment, confidence = max(probability_map.items(), key=lambda item: item[1])
                results.append(
                    AnalysisResult(
                        text="" if original_text is None else str(original_text),
                        sentiment=sentiment,
                        confidence=float(confidence),
                        probabilities=probability_map,
                        source=source_values[start + offset],
                    )
                )
            if progress_callback is not None:
                progress_callback(end, total)
        return results

    def _default_batch_size(self) -> int:
        value = (self.device_name or "CPU").strip()
        if value == "CPU":
            return 16
        return 32

    def _ensure_model(self):
        if self._model is not None and self._tokenizer is not None and self._torch_device is not None:
            return self._model, self._tokenizer, self._torch_device

        if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise TransformerLoadError(
                "Не установлены зависимости torch/transformers. Выполните: pip install -r requirements.txt"
            )
        if not self.model_name:
            raise TransformerLoadError("Не выбрана модель. Укажите локальную папку с обученной моделью.")

        local_only = os.getenv("SENTIMENT_ALLOW_MODEL_DOWNLOAD", "").strip() != "1"
        model_path = Path(self.model_name)
        if local_only and not model_path.exists():
            raise TransformerLoadError(
                "Модель не найдена локально. Укажите путь к папке с моделью или задайте "
                "SENTIMENT_ALLOW_MODEL_DOWNLOAD=1 для разовой загрузки."
            )

        load_name = str(model_path.resolve()) if model_path.exists() else self.model_name
        try:
            tokenizer = AutoTokenizer.from_pretrained(load_name, local_files_only=local_only)
            model = AutoModelForSequenceClassification.from_pretrained(load_name, local_files_only=local_only)
        except Exception as exc:  # pragma: no cover - depends on external libraries/files
            raise TransformerLoadError(f"Не удалось загрузить transformer-модель '{self.model_name}': {exc}") from exc

        torch_device = self._resolve_device()
        model.to(torch_device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._torch_device = torch_device
        return model, tokenizer, torch_device

    def _resolve_device(self):
        assert torch is not None
        value = (self.device_name or "CPU").strip()
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if value.startswith("CUDA"):
            parts = value.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return torch.device(f"cuda:{parts[1]}")
            return torch.device("cuda:0")
        if value == "Auto":
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _ordered_labels(self, model: object, count: int) -> list[str]:
        id2label = getattr(getattr(model, "config", None), "id2label", None) or {}
        labels: list[str] = []
        for index in range(count):
            raw_label = str(id2label.get(index, f"LABEL_{index}"))
            labels.append(self._normalize_label(raw_label))
        return labels

    @staticmethod
    def _normalize_sources(sources: list[object] | None, size: int) -> list[str]:
        if not sources:
            return [""] * size
        normalized = ["" if value is None else str(value) for value in sources[:size]]
        if len(normalized) < size:
            normalized.extend([""] * (size - len(normalized)))
        return normalized

    def _normalize_label(self, label: str) -> str:
        normalized = normalize_comparable_label(label)
        if normalized in CANONICAL_TO_RU:
            return CANONICAL_TO_RU[normalized]
        return normalized or "Класс"


def inspect_model_schema(model_name: str) -> ModelSchema:
    """Read the label schema from a local Hugging Face model folder."""
    path = Path(model_name)
    labels: list[str] = []
    if path.exists():
        config_path = path / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                config = {}
            id2label = config.get("id2label") or {}
            if isinstance(id2label, dict):
                for key, value in sorted(id2label.items(), key=lambda item: int(str(item[0])) if str(item[0]).isdigit() else str(item[0])):
                    label = str(value).strip()
                    if label:
                        labels.append(label)
    return ModelSchema(model_name=model_name, labels=tuple(labels))
