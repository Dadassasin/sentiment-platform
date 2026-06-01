"""Supervised fine-tuning for transformer sentiment classifiers."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

DEFAULT_HF_HOME = Path.cwd() / "models" / "huggingface"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover - dependency guard
    torch = None
    DataLoader = None
    Dataset = object
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


CANONICAL_LABELS = ("negative", "neutral", "positive")
ID_TO_RU_LABEL = {
    "negative": "Отрицательная",
    "neutral": "Нейтральная",
    "positive": "Положительная",
}

POSITIVE_LABEL_ALIASES = {
    "positive",
    "positiv",
    "pos",
    "+1",
    "1",
    "положительная",
    "положительный",
    "позитив",
    "позитивная",
    "позитивный",
}
NEUTRAL_LABEL_ALIASES = {
    "neutral",
    "neut",
    "neu",
    "0",
    "нейтральная",
    "нейтральный",
    "нейтрально",
}
NEGATIVE_LABEL_ALIASES = {
    "negative",
    "neg",
    "-1",
    "2",
    "отрицательная",
    "отрицательный",
    "негатив",
    "негативная",
    "негативный",
}


@dataclass(slots=True)
class TrainConfig:
    model_name: str
    output_dir: Path
    experiment_name: str = "sentiment_experiment"
    run_description: str = ""
    epochs: int = 2
    batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    validation_split: float = 0.2
    test_split: float = 0.1
    stratify_split: bool = True
    split_seed: int = 42
    seed: int = 42
    max_length: int = 256
    padding: str = "max_length"
    truncation: bool = True
    truncation_strategy: str = "longest_first"
    use_fast_tokenizer: bool = True
    pad_to_multiple_of: int | None = 8
    device: str = "CPU"
    local_files_only: bool = True
    trust_remote_code: bool = False
    ignore_mismatched_sizes: bool = True
    problem_type: str = "single_label_classification"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    optimizer: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    label_smoothing_factor: float = 0.0
    class_weights: str = "none"
    freeze_base_model: bool = False
    freeze_embeddings: bool = False
    freeze_encoder_layers: int = 0
    train_classifier_only: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    drop_duplicates: bool = True
    max_samples: int = 0
    shuffle_dataset: bool = True
    early_stopping: bool = True
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.0
    metric_for_best_model: str = "macro_f1"
    save_predictions: bool = True
    save_training_config: bool = True


@dataclass(slots=True)
class TrainResult:
    output_dir: Path
    accuracy: float
    macro_f1: float
    train_size: int
    validation_size: int
    label_to_id: dict[str, int]
    validation_source: str = "split"
    test_size: int = 0
    test_accuracy: float = 0.0
    test_macro_f1: float = 0.0
    test_source: str = "none"


class TrainingError(RuntimeError):
    """Raised when supervised fine-tuning cannot be completed."""


class TextClassificationDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def train_transformer_classifier(
    texts: list[str],
    labels: list[object],
    config: TrainConfig,
    progress: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    val_texts: list[str] | None = None,
    val_labels: list[object] | None = None,
    test_texts: list[str] | None = None,
    test_labels: list[object] | None = None,
) -> TrainResult:
    """Fine-tune a sequence-classification transformer and save it locally.

    Если val_texts/val_labels переданы - они используются как validation вместо
    отрезания процента от train. Если переданы test_texts/test_labels - после
    обучения выполняется held-out оценка и метрики сохраняются вместе с моделью.
    """

    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise TrainingError("Не установлены зависимости torch/transformers. Выполните: pip install -r requirements.txt")

    def cancelled() -> bool:
        return should_stop is not None and should_stop()

    set_seed(config.seed)
    if cancelled():
        raise TrainingError("Обучение прервано пользователем.")
    clean_rows = prepare_rows(texts, labels, config)
    if len(clean_rows) < 6:
        raise TrainingError("Для обучения нужно минимум 6 непустых размеченных текстов.")

    has_external_val = val_texts is not None and val_labels is not None and len(val_texts) > 0
    has_external_test = test_texts is not None and test_labels is not None and len(test_texts) > 0

    clean_train_texts = [row[0] for row in clean_rows]
    normalized_train_labels = normalize_labels([row[1] for row in clean_rows])

    all_normalized_labels = list(normalized_train_labels)
    normalized_val_labels: list[str] = []
    if has_external_val:
        normalized_val_labels = normalize_labels(list(val_labels))
        all_normalized_labels.extend(normalized_val_labels)
    normalized_test_labels: list[str] = []
    if has_external_test:
        normalized_test_labels = normalize_labels(list(test_labels))
        all_normalized_labels.extend(normalized_test_labels)

    label_to_id = build_label_mapping(all_normalized_labels)
    train_y_all = [label_to_id[label] for label in normalized_train_labels]

    if len(label_to_id) < 2:
        raise TrainingError("В колонке меток должен быть минимум 2 разных класса.")

    if has_external_test:
        test_texts_clean = [str(text).strip() for text in test_texts]
        test_y = [label_to_id[label] for label in normalized_test_labels]
        test_source = "external"
        train_pool_texts = clean_train_texts
        train_pool_y = train_y_all
    elif config.test_split > 0:
        train_pool_texts, test_texts_clean, train_pool_y, test_y = split_dataset(
            clean_train_texts,
            train_y_all,
            config.test_split,
            config.split_seed,
            config.stratify_split,
        )
        test_source = "split"
    else:
        test_texts_clean = []
        test_y = []
        test_source = "none"
        train_pool_texts = clean_train_texts
        train_pool_y = train_y_all

    if has_external_val:
        train_texts_split = train_pool_texts
        train_y = train_pool_y
        val_texts_split = [str(text).strip() for text in val_texts]
        val_y = [label_to_id[label] for label in normalized_val_labels]
        validation_source = "external"
    else:
        train_texts_split, val_texts_split, train_y, val_y = split_dataset(
            train_pool_texts,
            train_pool_y,
            config.validation_split,
            config.split_seed + 1,
            config.stratify_split,
        )
        validation_source = "split"

    # keep variable name compatible with downstream code
    train_texts = train_texts_split
    val_texts = val_texts_split
    local_only = config.local_files_only and os.getenv("SENTIMENT_ALLOW_MODEL_DOWNLOAD", "").strip() != "1"

    validation_label = "файл" if validation_source == "external" else "split"
    test_label = "файл" if test_source == "external" else test_source
    emit(progress, f"Фактическая выборка: train={len(train_texts)}, validation={len(val_texts)} ({validation_label})")
    if test_texts_clean:
        emit(progress, f"Test: {len(test_texts_clean)} строк ({test_label})")

    model_source = "локальный кэш" if local_only else "Hugging Face Hub или локальный кэш"
    emit(progress, f"Загрузка базовой модели: {config.model_name}")
    emit(progress, f"Источник модели: {model_source}")
    if cancelled():
        raise TrainingError("Обучение прервано пользователем.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            local_files_only=local_only,
            use_fast=config.use_fast_tokenizer,
            trust_remote_code=config.trust_remote_code,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=len(label_to_id),
            local_files_only=local_only,
            ignore_mismatched_sizes=config.ignore_mismatched_sizes,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as exc:
        hint = (
            "Снимите флажок 'Только локальные файлы' или укажите локальный путь к модели."
            if local_only
            else "Проверьте подключение к интернету или укажите локальный путь к уже скачанной модели."
        )
        raise TrainingError(
            f"Не удалось загрузить базовую модель '{config.model_name}'. "
            f"{hint}"
        ) from exc

    id_to_label = {index: label for label, index in label_to_id.items()}
    model.config.label2id = label_to_id
    model.config.id2label = id_to_label
    model.config.problem_type = config.problem_type
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    freeze_model_parts(model, config)

    device = resolve_device(config.device)
    model.to(device)

    emit(progress, "Токенизация данных")
    if cancelled():
        raise TrainingError("Обучение прервано пользователем.")
    train_dataset = make_dataset(tokenizer, train_texts, train_y, config)
    val_dataset = make_dataset(tokenizer, val_texts, val_y, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory and device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory and device.type == "cuda",
    )

    optimizer = build_optimizer(model, config)
    total_update_steps = max(1, math.ceil(len(train_loader) / max(config.gradient_accumulation_steps, 1)) * config.epochs)
    scheduler = build_scheduler(optimizer, config, total_update_steps)
    loss_fn = build_loss_fn(train_y, len(label_to_id), config, device)
    use_amp = config.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_accuracy = 0.0
    best_f1 = 0.0
    peak_accuracy = 0.0
    peak_f1 = 0.0
    best_metric = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    stale_evals = 0

    for epoch in range(1, config.epochs + 1):
        if cancelled():
            raise TrainingError("Обучение прервано пользователем.")
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            if cancelled():
                raise TrainingError("Обучение прервано пользователем.")
            batch = {key: value.to(device) for key, value in batch.items()}
            labels_tensor = batch.pop("labels")
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)
                loss = loss_fn(outputs.logits, labels_tensor)
                scaled_loss = loss / max(config.gradient_accumulation_steps, 1)
            scaler.scale(scaled_loss).backward()
            if step % max(config.gradient_accumulation_steps, 1) == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after_step = scaler.get_scale()
                if not use_amp or scale_after_step >= scale_before_step:
                    scheduler.step()
                optimizer.zero_grad()
            total_loss += float(loss.detach().cpu())

        if cancelled():
            raise TrainingError("Обучение прервано пользователем.")
        accuracy, macro_f1 = evaluate(model, val_loader, device)
        peak_accuracy = max(peak_accuracy, accuracy)
        peak_f1 = max(peak_f1, macro_f1)
        current_metric = macro_f1 if config.metric_for_best_model == "macro_f1" else accuracy
        if current_metric > best_metric + config.early_stopping_threshold:
            best_metric = current_metric
            stale_evals = 0
            best_accuracy = accuracy
            best_f1 = macro_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_evals += 1
        avg_loss = total_loss / max(len(train_loader), 1)
        emit(progress, f"Эпоха {epoch}/{config.epochs}: loss={avg_loss:.4f}, accuracy={accuracy:.3f}, macro_f1={macro_f1:.3f}")
        if config.early_stopping and stale_evals >= config.early_stopping_patience:
            emit(progress, f"Ранняя остановка: {config.metric_for_best_model} не улучшалась {stale_evals} проверок.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if cancelled():
        raise TrainingError("Обучение прервано пользователем.")

    test_accuracy = 0.0
    test_f1 = 0.0
    test_dataset = None
    test_loader = None
    if test_texts_clean:
        emit(progress, f"Held-out оценка на test ({len(test_texts_clean)} строк)")
        test_dataset = make_dataset(tokenizer, test_texts_clean, test_y, config)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.eval_batch_size,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory and device.type == "cuda",
        )
        test_accuracy, test_f1 = evaluate(model, test_loader, device)
        emit(progress, f"Test: accuracy={test_accuracy:.3f}, macro_f1={test_f1:.3f}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(config.output_dir)
    model.save_pretrained(config.output_dir)
    metadata = {
        "base_model": config.model_name,
        "accuracy": best_accuracy,
        "macro_f1": best_f1,
        "peak_accuracy": peak_accuracy,
        "peak_macro_f1": peak_f1,
        "train_size": len(train_texts),
        "validation_size": len(val_texts),
        "validation_source": validation_source,
        "test_size": len(test_texts_clean),
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_f1,
        "test_source": test_source,
        "label_to_id": label_to_id,
        "id_to_ru_label": {str(label_to_id[key]): ID_TO_RU_LABEL.get(key, key) for key in label_to_id},
        "experiment_config": serialize_config(config),
    }
    (config.output_dir / "training_metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if config.save_training_config:
        (config.output_dir / "training_config.json").write_text(
            json.dumps(serialize_config(config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if config.save_predictions:
        predictions = predict_label_ids(model, val_loader, device)
        prediction_rows = [
            {
                "text": text,
                "true_label_id": target,
                "predicted_label_id": prediction,
                "true_label": id_to_label.get(target, str(target)),
                "predicted_label": id_to_label.get(prediction, str(prediction)),
            }
            for text, target, prediction in zip(val_texts, val_y, predictions, strict=False)
        ]
        (config.output_dir / "validation_predictions.json").write_text(
            json.dumps(prediction_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if test_texts_clean and test_loader is not None:
            test_predictions = predict_label_ids(model, test_loader, device)
            test_prediction_rows = [
                {
                    "text": text,
                    "true_label_id": target,
                    "predicted_label_id": prediction,
                    "true_label": id_to_label.get(target, str(target)),
                    "predicted_label": id_to_label.get(prediction, str(prediction)),
                }
                for text, target, prediction in zip(test_texts_clean, test_y, test_predictions, strict=False)
            ]
            (config.output_dir / "test_predictions.json").write_text(
                json.dumps(test_prediction_rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    emit(progress, f"Модель сохранена: {config.output_dir}")

    return TrainResult(
        output_dir=config.output_dir,
        accuracy=best_accuracy,
        macro_f1=best_f1,
        train_size=len(train_texts),
        validation_size=len(val_texts),
        label_to_id=label_to_id,
        validation_source=validation_source,
        test_size=len(test_texts_clean),
        test_accuracy=test_accuracy,
        test_macro_f1=test_f1,
        test_source=test_source,
    )


def normalize_labels(labels: list[object]) -> list[str]:
    raw_values = [str(label).strip() for label in labels]
    if not raw_values:
        raise TrainingError("Колонка меток пуста.")

    normalized: list[str] = []
    for value in raw_values:
        normalized.append(canonicalize_label(value))
    return normalized


def build_label_mapping(labels: list[str]) -> dict[str, int]:
    unique_labels = set(labels)
    present = [label for label in CANONICAL_LABELS if label in unique_labels]
    present.extend(sorted(label for label in unique_labels if label not in set(CANONICAL_LABELS)))
    return {label: index for index, label in enumerate(present)}


def canonicalize_label(value: str) -> str:
    mapped = map_label(value)
    return mapped if mapped is not None else str(value).strip()


def map_label(value: str) -> str | None:
    raw_label = value.casefold().strip()
    compact_label = raw_label.replace("-", "_").replace(" ", "_")
    if raw_label in POSITIVE_LABEL_ALIASES or compact_label in POSITIVE_LABEL_ALIASES:
        return "positive"
    if raw_label in NEUTRAL_LABEL_ALIASES or compact_label in NEUTRAL_LABEL_ALIASES:
        return "neutral"
    if raw_label in NEGATIVE_LABEL_ALIASES or compact_label in NEGATIVE_LABEL_ALIASES:
        return "negative"
    return None


def prepare_rows(texts: list[str], labels: list[object], config: TrainConfig) -> list[tuple[str, object]]:
    rows = [(str(text).strip(), label) for text, label in zip(texts, labels, strict=False) if str(text).strip()]
    if config.drop_duplicates:
        seen: set[str] = set()
        deduplicated: list[tuple[str, object]] = []
        for text, label in rows:
            if text in seen:
                continue
            seen.add(text)
            deduplicated.append((text, label))
        rows = deduplicated
    if config.shuffle_dataset:
        random.Random(config.seed).shuffle(rows)
    if config.max_samples > 0:
        rows = rows[: config.max_samples]
    return rows


def split_dataset(
    texts: list[str],
    labels: list[int],
    validation_split: float,
    seed: int,
    stratify: bool,
) -> tuple[list[str], list[str], list[int], list[int]]:
    if stratify:
        return stratified_split_dataset(texts, labels, validation_split, seed)

    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, int(len(indices) * validation_split))
    validation_indices = set(indices[:validation_size])
    train_texts: list[str] = []
    validation_texts: list[str] = []
    train_labels: list[int] = []
    validation_labels: list[int] = []

    for index, (text, label) in enumerate(zip(texts, labels, strict=False)):
        if index in validation_indices:
            validation_texts.append(text)
            validation_labels.append(label)
        else:
            train_texts.append(text)
            train_labels.append(label)

    return train_texts, validation_texts, train_labels, validation_labels


def stratified_split_dataset(
    texts: list[str],
    labels: list[int],
    validation_split: float,
    seed: int,
) -> tuple[list[str], list[str], list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)

    validation_indices: set[int] = set()
    for indices in by_label.values():
        rng.shuffle(indices)
        validation_size = max(1, int(len(indices) * validation_split)) if len(indices) > 1 else 0
        validation_indices.update(indices[:validation_size])

    if not validation_indices:
        validation_indices.add(rng.randrange(len(texts)))

    train_texts: list[str] = []
    validation_texts: list[str] = []
    train_labels: list[int] = []
    validation_labels: list[int] = []
    for index, (text, label) in enumerate(zip(texts, labels, strict=False)):
        if index in validation_indices:
            validation_texts.append(text)
            validation_labels.append(label)
        else:
            train_texts.append(text)
            train_labels.append(label)
    return train_texts, validation_texts, train_labels, validation_labels


def make_dataset(tokenizer: object, texts: list[str], labels: list[int], config: TrainConfig) -> TextClassificationDataset:
    padding: bool | str = config.padding
    if padding == "false":
        padding = False
    tokenizer_kwargs = {
        "padding": padding,
        "truncation": config.truncation,
        "max_length": config.max_length,
        "pad_to_multiple_of": config.pad_to_multiple_of if padding else None,
        "return_tensors": "pt",
    }
    # New tokenizers backends reject truncation_strategy for encode_plus/batch encode.
    # For our single-sequence classification inputs truncation+max_length is sufficient.
    encodings = tokenizer(texts, **tokenizer_kwargs)
    return TextClassificationDataset(encodings, labels)


def build_optimizer(model: object, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = config.warmup_steps or int(total_steps * config.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        if config.lr_scheduler_type == "constant":
            return 1.0
        if config.lr_scheduler_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_loss_fn(labels: list[int], num_labels: int, config: TrainConfig, device: torch.device) -> torch.nn.CrossEntropyLoss:
    weights = None
    if config.class_weights == "balanced":
        counts = {label: labels.count(label) for label in range(num_labels)}
        total = sum(counts.values())
        weights = torch.tensor(
            [total / max(num_labels * counts.get(label, 1), 1) for label in range(num_labels)],
            dtype=torch.float,
            device=device,
        )
    return torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing_factor)


def freeze_model_parts(model: object, config: TrainConfig) -> None:
    if config.freeze_base_model or config.train_classifier_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(part in name for part in ("classifier", "score", "pre_classifier"))

    if config.freeze_embeddings:
        embeddings = getattr(model, "embeddings", None) or getattr(getattr(model, "bert", None), "embeddings", None)
        if embeddings is not None:
            for parameter in embeddings.parameters():
                parameter.requires_grad = False

    if config.freeze_encoder_layers > 0:
        encoder = getattr(model, "encoder", None) or getattr(getattr(model, "bert", None), "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is not None:
            for layer in list(layers)[: config.freeze_encoder_layers]:
                for parameter in layer.parameters():
                    parameter.requires_grad = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def serialize_config(config: TrainConfig) -> dict[str, object]:
    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    return data


def evaluate(model: object, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].detach().cpu().numpy().tolist()
            inputs = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).detach().cpu().numpy().tolist()
            predictions.extend(preds)
            targets.extend(labels)

    if not targets:
        return 0.0, 0.0
    return accuracy(targets, predictions), macro_f1(targets, predictions)


def predict_label_ids(model: object, loader: DataLoader, device: torch.device) -> list[int]:
    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            outputs = model(**inputs)
            predictions.extend(torch.argmax(outputs.logits, dim=1).detach().cpu().numpy().tolist())
    return predictions


def accuracy(targets: list[int], predictions: list[int]) -> float:
    if not targets:
        return 0.0
    return sum(target == prediction for target, prediction in zip(targets, predictions, strict=False)) / len(targets)


def macro_f1(targets: list[int], predictions: list[int]) -> float:
    labels = sorted(set(targets) | set(predictions))
    if not labels:
        return 0.0
    scores: list[float] = []
    for label in labels:
        tp = sum(target == label and prediction == label for target, prediction in zip(targets, predictions, strict=False))
        fp = sum(target != label and prediction == label for target, prediction in zip(targets, predictions, strict=False))
        fn = sum(target == label and prediction != label for target, prediction in zip(targets, predictions, strict=False))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("CUDA") or device_name == "Auto":
        if torch.cuda.is_available():
            if device_name.startswith("CUDA"):
                parts = device_name.split()
                if len(parts) > 1 and parts[1].isdigit():
                    return torch.device(f"cuda:{parts[1]}")
            return torch.device("cuda:0")
    return torch.device("cpu")


def emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
