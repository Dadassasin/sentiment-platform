from __future__ import annotations

import json
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QEvent, QThread, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QIcon
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from app.active_learning import UncertainExample, select_uncertain_examples
from app.models.sentiment import (
    AnalysisResult,
    MODEL_PROFILES,
    TRAINING_BASE_MODELS,
    TOKENIZER_VOCAB_FILES,
    inspect_model_schema,
    SentimentAnalyzer,
    TransformerLoadError,
    WEIGHT_FILES,
)
from app.models.comparison import compare_models
from app.monitoring import build_drift_report
from app.preprocessing import PreprocessingOptions
from app.reports import export_html_report
from app.training.experiments import class_distribution, comparison_rows, summarize_results
from app.training.transformer_trainer import TrainConfig, TrainResult, TrainingError, train_transformer_classifier

def resource_path(relative_path: str) -> str:
    import sys
    base_path = getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)
    return str(Path(base_path) / relative_path)

POSITIVE = "Положительная"
NEUTRAL = "Нейтральная"
NEGATIVE = "Отрицательная"

TARGET_CLASSES = [POSITIVE, NEUTRAL, NEGATIVE]
KEEP_ORIGINAL = "Оставить исходную"
EXCLUDE_LABEL = "Исключить"
ORIGINAL_SCHEME = "Оставить выбранные исходные классы"
SENTIMENT_SCHEME = "Свести к тональности: положительная / нейтральная / отрицательная"

SENTIMENT_COLORS = {
    POSITIVE: "#2563eb",
    NEUTRAL: "#d97706",
    NEGATIVE: "#dc2626",
}

ACTIVE_LEARNING_THRESHOLD = 0.60

# Dense desktop design tokens, all spacing values use a 4px grid.
SPACE_1 = 4
SPACE_2 = 8
SPACE_3 = 12
SPACE_4 = 16


TRAINING_FIELD_HELP = {
    "Название эксперимента": "Имя запуска для логов, папок и сравнения результатов между экспериментами.",
    "Папка сохранения": "Куда будут записаны модель, tokenizer, метрики, training_config.json и predictions.",
    "Seed": "Фиксирует случайность разбиения и обучения, чтобы запуск можно было повторить.",
    "Описание": "Заметка о гипотезе, датасете или отличиях этого запуска от других.",
    "Колонка текста": "Столбец датасета, из которого берутся тексты для дообучения модели.",
    "Колонка метки": "Столбец с исходными метками; лишние значения настраиваются во вкладке Метки.",
    "Максимум строк": "Ограничивает размер обучающего набора для быстрых пробных запусков; 0 значит без ограничения.",
    "Очистка": "Удаляет дубли и перемешивает строки перед разбиением на train/validation.",
    "Validation split": "Доля данных, отложенная для проверки качества после каждой эпохи.",
    "Test split": "Доля данных, отложенная для независимой оценки после обучения. Применяется только если отдельная тестовая выборка не загружена. 0 - не использовать тестовую выборку.",
    "Split seed": "Отдельный seed для разбиения данных на train и validation.",
    "Стратегия": "Стратификация сохраняет похожее распределение классов в train и validation.",
    "Веса классов": "Balanced повышает вклад редких классов, если данные несбалансированы.",
    "Лучшая модель по": "Метрика, по которой выбирается лучшая сохранённая версия модели и выполняется ранняя остановка.",
    "Базовая модель": "Pretrained checkpoint, от которого начинается дообучение классификатора.",
    "Тип задачи": "Для текущего sentiment-classifier используется single-label classification.",
    "Загрузка": "По умолчанию Hugging Face модели скачиваются в локальный кэш. Включайте локальный режим только для офлайн-запуска.",
    "Заморозка": "Позволяет обучать только голову или заморозить embeddings на маленьком датасете.",
    "Первые N слоёв": "Замораживает первые encoder-слои, снижая расход памяти и риск переобучения.",
    "Max length": "Максимальное число токенов на текст; больше длина значит медленнее и больше памяти.",
    "Padding": "max_length даёт одинаковую длину батча; longest обычно экономит память.",
    "Truncation": "Обрезает слишком длинные тексты до Max length.",
    "Truncation strategy": "Определяет, какую часть текста обрезать при превышении Max length.",
    "Tokenizer": "Fast tokenizer обычно быстрее и подходит для большинства Hugging Face моделей.",
    "Pad to multiple of": "Выравнивает длину последовательностей, часто полезно для GPU/FP16; 0 отключает.",
    "Эпохи": "Сколько полных проходов по train-данным выполнит обучение.",
    "Train batch": "Сколько текстов обрабатывается за один training step на выбранном устройстве.",
    "Eval batch": "Размер батча для проверки качества; может быть больше train batch.",
    "Learning rate": "Скорость обновления весов; для BERT/RuBERT обычно начинают с 2e-5.",
    "Weight decay": "L2-регуляризация AdamW, помогает не переобучаться.",
    "Gradient accumulation": "Накапливает градиенты несколько шагов, имитируя больший batch при малой памяти.",
    "Gradient clipping": "Ограничивает норму градиента и защищает обучение от резких скачков.",
    "Оптимизатор": "Алгоритм обновления весов; AdamW является стандартом для трансформеров.",
    "Adam beta1": "Параметр сглаживания первого момента в AdamW.",
    "Adam beta2": "Параметр сглаживания второго момента в AdamW.",
    "Adam epsilon": "Численная стабилизация AdamW, обычно 1e-8.",
    "Scheduler": "Как learning rate меняется во время обучения.",
    "Warmup ratio": "Доля шагов, за которую learning rate плавно растёт от нуля.",
    "Warmup steps": "Абсолютное число warmup-шагов; если больше 0, важнее Warmup ratio.",
    "Label smoothing": "Смягчает целевые классы, полезно при шумной разметке.",
    "Early stopping": "Останавливает обучение, если выбранная метрика перестала улучшаться.",
    "Patience": "Сколько проверок подряд ждать улучшения перед остановкой.",
    "Устройство": "Где обучать модель: Auto, CPU или CUDA.",
    "Precision": "FP16 ускоряет CUDA-обучение; gradient checkpointing экономит память ценой скорости.",
    "Dataloader workers": "Число процессов подготовки батчей; 0 надёжнее на Windows.",
    "Memory": "Pin memory ускоряет передачу батчей на CUDA.",
    "Сохранение": "Сохраняет параметры запуска и предсказания на валидационной выборке для анализа ошибок.",
    "Формат меток": "Одна метка читает ячейку целиком; список разбирает значения вида (3, 5, 7).",
    "Если меток несколько": "Что делать со строкой, где в одной ячейке найдено несколько меток.",
    "Схема обучения": "Оставить исходные классы для эмоций/тем/категорий или явно свести датасет к 3 sentiment-классам.",
}


POSITIVE_ALIASES = {
    "positive",
    "positiv",
    "pos",
    "1",
    "+1",
    "положительная",
    "положительный",
    "позитив",
    "позитивная",
    "позитивный",
    "good",
    "like",
    "liked",
    "joy",
    "happy",
}

NEUTRAL_ALIASES = {
    "neutral",
    "neut",
    "neu",
    "0",
    "нейтральная",
    "нейтральный",
    "нейтрально",
    "нормально",
    "normal",
    "mixed",
}

NEGATIVE_ALIASES = {
    "negative",
    "neg",
    "-1",
    "2",
    "негатив",
    "негативная",
    "негативный",
    "отрицательная",
    "отрицательный",
    "bad",
    "dislike",
    "sadness",
    "anger",
    "angry",
}

EXCLUDE_ALIASES = {
    "skip",
    "speech",
    "other",
    "irrelevant",
    "unknown",
    "none",
    "nan",
    "null",
    "undefined",
    "trash",
    "junk",
    "spam",
    "неизвестно",
    "другое",
    "прочее",
    "пропустить",
}


@dataclass(frozen=True)
class LabelPrepareResult:
    texts: list[str]
    labels: list[str]
    used_rows: int
    excluded_rows: int
    class_counts: Counter[str]


class TrainingTabWidget(QWidget):
    """Equal-width tab control for the training workbench."""

    def __init__(self) -> None:
        super().__init__()
        self.buttons: list[QPushButton] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tab_row = QWidget()
        self.tab_row.setObjectName("trainingTabRow")
        self.tab_layout = QHBoxLayout(self.tab_row)
        self.tab_layout.setContentsMargins(0, 0, 0, 0)
        self.tab_layout.setSpacing(0)

        self.stack = QStackedWidget()
        self.stack.setObjectName("trainingTabStack")
        layout.addWidget(self.tab_row)
        layout.addWidget(self.stack, 1)

    def addTab(self, widget: QWidget, title: str) -> int:
        index = self.stack.addWidget(widget)
        button = QPushButton(title)
        button.setObjectName("trainingTabButton")
        button.setCheckable(True)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.clicked.connect(lambda checked=False, page=index: self.setCurrentIndex(page))
        self.buttons.append(button)
        self.tab_layout.addWidget(button, 1)
        if index == 0:
            button.setChecked(True)
        return index

    def setCurrentIndex(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for button_index, button in enumerate(self.buttons):
            button.setChecked(button_index == index)

    def setTabToolTip(self, index: int, tooltip: str) -> None:
        if 0 <= index < len(self.buttons):
            self.buttons[index].setToolTip(tooltip)

    def count(self) -> int:
        return self.stack.count()


class CenterStatusBar(QStatusBar):
    def __init__(self) -> None:
        super().__init__()
        self._message_label = QLabel("")
        self._message_label.setObjectName("statusMessageLabel")
        self._message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.clearMessage)
        self.addWidget(self._message_label, 1)

    def showMessage(self, message: str, timeout: int = 0) -> None:
        self._timer.stop()
        self._message_label.setText(message)
        self.messageChanged.emit(message)
        if timeout > 0:
            self._timer.start(timeout)

    def clearMessage(self) -> None:
        self._timer.stop()
        self._message_label.clear()
        self.messageChanged.emit("")

    def currentMessage(self) -> str:
        return self._message_label.text()


def format_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def result_score_labels(results: list[AnalysisResult]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for result in results:
        for label in result.probabilities:
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def format_probabilities(probabilities: dict[str, float], limit: int = 6) -> str:
    if not probabilities:
        return "Вероятности появятся после анализа"
    items = list(probabilities.items())
    visible = items[:limit]
    text = " · ".join(f"{label} {score:.2f}" for label, score in visible)
    hidden = len(items) - len(visible)
    if hidden > 0:
        text += f" · ещё {hidden}"
    return text


def compact_score_label(label: str) -> str:
    mapping = {
        POSITIVE: "Полож.",
        NEUTRAL: "Нейтр.",
        NEGATIVE: "Отриц.",
    }
    return mapping.get(label, label)


def device_labels() -> list[str]:
    labels = ["Auto", "CPU"]
    try:
        import torch
    except ImportError:
        return labels

    if not torch.cuda.is_available():
        return labels

    device_count = torch.cuda.device_count()
    for index in range(device_count):
        name = torch.cuda.get_device_name(index)
        label = f"CUDA ({name})" if device_count == 1 else f"CUDA {index} ({name})"
        labels.append(label)
    return labels


def label_key(value: object) -> str:
    return str(value).strip().strip("'\"").casefold()


def parse_label_tokens(value: object, parse_as_list: bool) -> list[str]:
    """Parse one label cell.

    If parse_as_list=False, the whole cell is treated as a single label.
    If parse_as_list=True, values like "(3,5,6)", "[positive, toxic]" or "positive;neutral"
    are split into separate label tokens.
    """
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    if not parse_as_list:
        return [text.strip().strip("'\"")]

    # remove common brackets around label lists
    text = text.strip().strip("[](){}")
    if not text:
        return []

    parts = re.split(r"[,;|]", text)
    tokens = [part.strip().strip("'\"") for part in parts]
    return [token for token in tokens if token]


def auto_target_for_label(raw_label: str) -> str:
    key = label_key(raw_label)
    if key in POSITIVE_ALIASES:
        return POSITIVE
    if key in NEUTRAL_ALIASES:
        return NEUTRAL
    if key in NEGATIVE_ALIASES:
        return NEGATIVE
    if key in EXCLUDE_ALIASES:
        return EXCLUDE_LABEL
    return KEEP_ORIGINAL


def make_combo(values: list[str], current: str, *, editable: bool = False) -> QComboBox:
    combo = QComboBox()
    combo.addItems(values)
    combo.setEditable(editable)
    combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    if current in values:
        combo.setCurrentText(current)
    elif editable and current:
        combo.setEditText(current)

    combo.setObjectName("tableCombo")
    combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    combo.setFixedHeight(32)
    combo.setMinimumWidth(0)
    combo.setMinimumContentsLength(18 if editable else 0)

    return combo


class Panel(QFrame):
    def __init__(self, title: str = "") -> None:
        super().__init__()
        self.setObjectName("panel")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        self.layout.setSpacing(SPACE_3)
        if title:
            label = QLabel(title)
            label.setObjectName("panelTitle")
            self.layout.addWidget(label)


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "0", subtitle: str = "") -> None:
        super().__init__()
        self.setObjectName("metricCard")
        self.setMinimumHeight(58)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE_3, SPACE_1, SPACE_3, SPACE_1)
        layout.setSpacing(SPACE_1)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("mutedLabel")
        self.subtitle_label.setWordWrap(True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.subtitle_label)

    def set_values(self, value: str, subtitle: str = "") -> None:
        self.value_label.setText(value)
        self.subtitle_label.setText(subtitle)


class ChartWidget(Panel):
    def __init__(self, title: str, min_height: int = 160) -> None:
        super().__init__(title)
        self.figure = Figure(figsize=(4, 2.2), tight_layout=True, facecolor="#ffffff")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(min_height)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.layout.addWidget(self.canvas, 1)

    def empty(self, message: str = "Нет данных") -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10, color="#6b7280")
        self.canvas.draw_idle()

    def draw_class_distribution(self, counts: Counter[str]) -> None:
        total = sum(counts.values())
        if total == 0:
            self.empty()
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        labels = [label for label in (POSITIVE, NEUTRAL, NEGATIVE) if counts.get(label, 0)]
        extra = [label for label in counts.keys() if label not in labels and counts.get(label, 0)]
        labels.extend(extra[:5])
        values = [counts[label] for label in labels]
        colors = [SENTIMENT_COLORS.get(label, "#6b7280") for label in labels]
        ax.pie(values, autopct="%1.1f%%", startangle=90, colors=colors, textprops={"fontsize": 8})
        ax.legend(
            [f"{label} ({counts[label]})" for label in labels],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=8,
            frameon=False,
        )
        ax.axis("equal")
        self.canvas.draw_idle()

    def draw_confidence_histogram(self, results: list[AnalysisResult]) -> None:
        if not results:
            self.empty()
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.hist([item.confidence for item in results], bins=16, color="#7aa5dc", edgecolor="white")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Уверенность", fontsize=8)
        ax.set_ylabel("Количество", fontsize=8)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        self.canvas.draw_idle()

    def draw_drift(self, results: list[AnalysisResult]) -> None:
        report = build_drift_report(results, batch_size=max(len(results) // 8, 25) if results else 25)
        if not report.points:
            self.empty()
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        batches = [point.batch for point in report.points]
        ax.plot(batches, [point.avg_confidence for point in report.points], marker="o", label="Уверенность")
        top_labels = Counter()
        for point in report.points:
            top_labels.update(point.class_shares)
        for label, _ in top_labels.most_common(3):
            ax.plot(
                batches,
                [point.class_shares.get(label, 0.0) for point in report.points],
                marker="o",
                label=label,
            )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Батч", fontsize=8)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.legend(fontsize=8, frameon=False)
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        self.canvas.draw_idle()

    def draw_training_result(self, result: TrainResult) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        groups = ["Validation"]
        accuracy = [result.accuracy]
        macro_f1 = [result.macro_f1]
        if result.test_size > 0:
            groups.append("Test")
            accuracy.append(result.test_accuracy)
            macro_f1.append(result.test_macro_f1)

        positions = range(len(groups))
        width = 0.34
        ax.bar([position - width / 2 for position in positions], accuracy, width, label="Accuracy", color="#7aa5dc")
        ax.bar([position + width / 2 for position in positions], macro_f1, width, label="Macro F1", color="#d97706")
        ax.set_ylim(0, 1)
        ax.set_xticks(list(positions), groups)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        self.canvas.draw_idle()

    def draw_grouped_bars(
        self,
        categories: list[str],
        series: list[tuple[str, list[float], str]],
        *,
        y_max: float | None = None,
        rotate_labels: int = 0,
    ) -> None:
        if not categories or not series:
            self.empty()
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        positions = list(range(len(categories)))
        series_count = len(series)
        width = 0.8 / max(series_count, 1)
        offset_base = (series_count - 1) / 2
        for index, (label, values, color) in enumerate(series):
            offset = (index - offset_base) * width
            ax.bar([position + offset for position in positions], values, width, label=label, color=color)
        if y_max is not None:
            ax.set_ylim(0, y_max)
        ax.set_xticks(positions, categories, rotation=rotate_labels, ha="right" if rotate_labels else "center")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        self.canvas.draw_idle()

    def draw_heatmap(self, labels: list[str], matrix: list[list[float]]) -> None:
        if not labels or not matrix:
            self.empty()
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.tick_params(labelsize=8)
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                text_color = "#ffffff" if value >= 0.55 else "#1f2937"
                ax.text(column_index, row_index, f"{value:.0%}", ha="center", va="center", fontsize=7, color=text_color)
        color_bar = self.figure.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
        color_bar.ax.tick_params(labelsize=8)
        self.canvas.draw_idle()


class TrainingThread(QThread):
    message = pyqtSignal(str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    canceled = pyqtSignal(str)

    def __init__(
        self,
        texts: list[str],
        labels: list[object],
        config: TrainConfig,
        val_texts: list[str] | None = None,
        val_labels: list[object] | None = None,
        test_texts: list[str] | None = None,
        test_labels: list[object] | None = None,
    ) -> None:
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.config = config
        self.val_texts = val_texts
        self.val_labels = val_labels
        self.test_texts = test_texts
        self.test_labels = test_labels
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def should_stop(self) -> bool:
        return self._stop_requested

    def run(self) -> None:
        try:
            result = train_transformer_classifier(
                self.texts,
                self.labels,
                self.config,
                self.message.emit,
                should_stop=self.should_stop,
                val_texts=self.val_texts,
                val_labels=self.val_labels,
                test_texts=self.test_texts,
                test_labels=self.test_labels,
            )
        except TrainingError as exc:
            message = str(exc)
            if self._stop_requested or "прервано пользователем" in message.casefold():
                self.canceled.emit(message)
            else:
                self.failed.emit(message)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(f"Ошибка обучения: {exc}")
        else:
            self.completed.emit(result)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Система анализа тональности текста")
        self.setWindowIcon(QIcon(resource_path("assets/icons/app_icon.ico")))
        self.resize(1280, 820)
        self.setMinimumSize(1100, 620)

        self.dataset_path: Path | None = None
        self.data_frame = pd.DataFrame()
        self.analysis_dataset_path: Path | None = None
        self.analysis_data_frame = pd.DataFrame()
        self.val_dataset_path: Path | None = None
        self.val_data_frame = pd.DataFrame()
        self.test_dataset_path: Path | None = None
        self.test_data_frame = pd.DataFrame()
        self.preview_source = "train"
        self.current_report_path: Path | None = None
        self.results: list[AnalysisResult] = []
        self.comparison_behavior_rows: list[object] = []
        self.comparison_quality_rows: list[object] = []
        self.event_log: list[tuple[str, str]] = []
        self.preview_offset = 0
        self.nav_buttons: list[QPushButton] = []
        self.training_thread: TrainingThread | None = None
        self.label_action_combos: dict[str, QComboBox] = {}
        self.label_target_edits: dict[str, QLineEdit] = {}
        self.label_count_by_token: Counter[str] = Counter()
        self._fill_width_tables: dict[int, QTableWidget] = {}
        self._fill_width_viewports: dict[int, QTableWidget] = {}
        self._table_resize_guard: set[int] = set()
        self._populating_profile_table = False
        self._pending_profile_rename_path: str | None = None
        self._pending_profile_rename_original_name: str = ""

        self._build_ui()
        self._apply_styles()
        self._set_initial_state()

    def _build_ui(self) -> None:
        self.setStatusBar(CenterStatusBar())
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_context_bar())

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_data_page())
        self.pages.addWidget(self._build_analysis_page())
        self.pages.addWidget(self._build_training_page())
        self.pages.addWidget(self._build_models_page())
        self.pages.addWidget(self._build_comparison_page())
        self.pages.addWidget(self._build_monitoring_page())
        self.pages.addWidget(self._build_reports_page())

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(self._build_sidebar())
        body_layout.addWidget(self.pages, 1)

        root_layout.addWidget(body, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Готово. Загрузите датасет для начала работы.")
        self.model_combo.currentTextChanged.connect(self._update_context_bar)
        self.train_device_combo.currentTextChanged.connect(self._update_context_bar)

    def _build_context_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("contextBar")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(SPACE_3, SPACE_2, SPACE_3, SPACE_2)
        layout.setSpacing(SPACE_3)

        self.context_dataset_label = self._add_context_field(
            layout,
            "Датасет:",
            "не загружен",
            object_name="contextValue",
            stretch=1,
            min_width=160,
        )
        layout.addWidget(self._context_separator())
        self.context_train_label = self._add_context_field(layout, "Train:", "0")
        layout.addWidget(self._context_separator())
        self.context_val_label = self._add_context_field(layout, "Val:", "-")
        layout.addWidget(self._context_separator())
        self.context_test_label = self._add_context_field(layout, "Test:", "-")
        layout.addWidget(self._context_separator())
        self.context_model_label = self._add_context_field(
            layout, "Модель:", "-", min_width=140
        )
        layout.addWidget(self._context_separator())
        self.context_device_label = self._add_context_field(layout, "Устройство:", "-")
        layout.addWidget(self._context_separator())

        self.status_label = QLabel("● Готово")
        self.status_label.setObjectName("statusChip")
        self.status_label.setProperty("state", "ready")
        self.status_label.setMinimumWidth(180)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        return bar

    def _add_context_field(
        self,
        layout: QHBoxLayout,
        title: str,
        value: str,
        object_name: str = "contextSmallValue",
        stretch: int = 0,
        min_width: int = 0,
    ) -> QLabel:
        title_label = QLabel(title)
        title_label.setObjectName("contextTitle")
        value_label = QLabel(value)
        value_label.setObjectName(object_name)
        if min_width:
            value_label.setMinimumWidth(min_width)
        layout.addWidget(title_label)
        if stretch:
            layout.addWidget(value_label, stretch)
        else:
            layout.addWidget(value_label)
        return value_label

    def _context_separator(self) -> QLabel:
        separator = QLabel("|")
        separator.setObjectName("contextSeparator")
        return separator

    def _build_sidebar(self) -> QWidget:
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(188)
        side.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(SPACE_2, SPACE_3, SPACE_2, SPACE_2)
        layout.setSpacing(SPACE_1)

        nav_title = QLabel("НАВИГАЦИЯ")
        nav_title.setObjectName("sidebarSection")
        layout.addWidget(nav_title)
        nav = ["Данные", "Анализ", "Обучение", "Модели", "Сравнение", "Мониторинг", "Отчёты"]
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        for index, title in enumerate(nav):
            button = QPushButton(title)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, page=index: self._switch_page(page))
            self.nav_group.addButton(button, index)
            self.nav_buttons.append(button)
            layout.addWidget(button)

        layout.addStretch(1)
        return side

    def _build_data_page(self) -> QWidget:
        page = self._page()
        self._add_page_header(
            page,
            "Данные",
            "Загрузите train, при необходимости отдельные validation и test, затем проверьте структуру таблицы.",
        )

        summary = Panel("Загрузка и подготовка данных")
        slots_grid = QGridLayout()
        slots_grid.setHorizontalSpacing(SPACE_2)
        slots_grid.setVerticalSpacing(SPACE_2)
        slots_grid.setColumnStretch(3, 1)

        self.train_load_button = QPushButton("Загрузить")
        self.train_load_button.setObjectName("primaryButton")
        self.train_load_button.clicked.connect(self.load_train_dataset)
        self.train_clear_button = QPushButton("Очистить")
        self.train_clear_button.clicked.connect(self.clear_train_dataset)
        self.train_status_label = QLabel("Файл не открыт. Train обязателен для анализа и обучения.")
        self.train_status_label.setObjectName("mutedLabel")
        self.train_status_label.setWordWrap(True)
        train_title = QLabel("Train dataset")
        train_title.setObjectName("formLabel")
        slots_grid.addWidget(train_title, 0, 0)
        slots_grid.addWidget(self.train_load_button, 0, 1)
        slots_grid.addWidget(self.train_clear_button, 0, 2)
        slots_grid.addWidget(self.train_status_label, 0, 3)

        self.val_load_button = QPushButton("Загрузить")
        self.val_load_button.clicked.connect(self.load_val_dataset)
        self.val_clear_button = QPushButton("Очистить")
        self.val_clear_button.clicked.connect(self.clear_val_dataset)
        self.val_status_label = QLabel("Не загружен. Validation будет отделена из train по проценту в настройках обучения.")
        self.val_status_label.setObjectName("mutedLabel")
        self.val_status_label.setWordWrap(True)
        val_title = QLabel("Validation dataset")
        val_title.setObjectName("formLabel")
        slots_grid.addWidget(val_title, 1, 0)
        slots_grid.addWidget(self.val_load_button, 1, 1)
        slots_grid.addWidget(self.val_clear_button, 1, 2)
        slots_grid.addWidget(self.val_status_label, 1, 3)

        self.test_load_button = QPushButton("Загрузить")
        self.test_load_button.clicked.connect(self.load_test_dataset)
        self.test_clear_button = QPushButton("Очистить")
        self.test_clear_button.clicked.connect(self.clear_test_dataset)
        self.test_status_label = QLabel("Не загружен. Test будет отделён из train по проценту в настройках обучения.")
        self.test_status_label.setObjectName("mutedLabel")
        self.test_status_label.setWordWrap(True)
        test_title = QLabel("Test dataset")
        test_title.setObjectName("formLabel")
        slots_grid.addWidget(test_title, 2, 0)
        slots_grid.addWidget(self.test_load_button, 2, 1)
        slots_grid.addWidget(self.test_clear_button, 2, 2)
        slots_grid.addWidget(self.test_status_label, 2, 3)

        summary.layout.addLayout(slots_grid)
        hybrid_hint = QLabel(
            "Колонки текста и метки и таблица соответствия меток применяются ко всем загруженным файлам."
        )
        hybrid_hint.setObjectName("mutedLabel")
        hybrid_hint.setWordWrap(True)
        summary.layout.addWidget(hybrid_hint)
        page.layout().addWidget(summary)

        data_metrics = QGridLayout()
        data_metrics.setHorizontalSpacing(SPACE_3)
        data_metrics.setVerticalSpacing(SPACE_3)
        self.train_rows_card = MetricCard("Train", "0", "строк")
        self.val_rows_card = MetricCard("Validation", "-", "не загружен")
        self.test_rows_card = MetricCard("Test", "-", "не загружен")
        self.dataset_columns_card = MetricCard("Колонки train", "0", "доступно для выбора")
        for column, card in enumerate((
            self.train_rows_card,
            self.val_rows_card,
            self.test_rows_card,
            self.dataset_columns_card,
        )):
            data_metrics.addWidget(card, 0, column)
            data_metrics.setColumnStretch(column, 1)
        page.layout().addLayout(data_metrics)

        preview = Panel("Предпросмотр данных")
        source_row = QHBoxLayout()
        source_row.setSpacing(SPACE_2)
        source_row.addWidget(QLabel("Выборка:"))
        self.preview_source_combo = QComboBox()
        self.preview_source_combo.addItem("Train", "train")
        self.preview_source_combo.addItem("Validation", "val")
        self.preview_source_combo.addItem("Test", "test")
        self.preview_source_combo.currentIndexChanged.connect(self._on_preview_source_changed)
        source_row.addWidget(self.preview_source_combo)
        source_row.addStretch(1)
        preview.layout.addLayout(source_row)

        preview_controls = QHBoxLayout()
        preview_controls.setSpacing(SPACE_2)
        self.preview_range_label = QLabel("Строки 0-0 из 0")
        self.preview_range_label.setObjectName("mutedLabel")

        self.preview_page_size_spin = QSpinBox()
        self.preview_page_size_spin.setRange(25, 5000)
        self.preview_page_size_spin.setSingleStep(25)
        self.preview_page_size_spin.setValue(200)
        self.preview_page_size_spin.setSuffix(" строк")
        self.preview_page_size_spin.setToolTip("Сколько строк показывать в предпросмотре за один раз")
        self.preview_page_size_spin.valueChanged.connect(self._reset_preview_page)

        self.preview_jump_spin = QSpinBox()
        self.preview_jump_spin.setRange(1, 1)
        self.preview_jump_spin.setPrefix("с ")
        self.preview_jump_spin.setSuffix(" строки")
        self.preview_jump_spin.setToolTip("Номер строки, с которой начать предпросмотр")

        jump_button = QPushButton("Перейти")
        jump_button.clicked.connect(self._jump_preview_page)
        self.preview_first_button = QPushButton("В начало")
        self.preview_first_button.clicked.connect(self._preview_first_page)
        self.preview_prev_button = QPushButton("Назад")
        self.preview_prev_button.clicked.connect(self._preview_prev_page)
        self.preview_next_button = QPushButton("Вперёд")
        self.preview_next_button.clicked.connect(self._preview_next_page)

        preview_controls.addWidget(self.preview_range_label)
        preview_controls.addStretch(1)
        preview_controls.addWidget(QLabel("Показать:"))
        preview_controls.addWidget(self.preview_page_size_spin)
        preview_controls.addWidget(self.preview_jump_spin)
        preview_controls.addWidget(jump_button)
        preview_controls.addWidget(self.preview_first_button)
        preview_controls.addWidget(self.preview_prev_button)
        preview_controls.addWidget(self.preview_next_button)
        preview.layout.addLayout(preview_controls)

        self.preview_table = QTableWidget(0, 0)
        self._configure_embedded_table(self.preview_table)
        self._enable_fill_width_columns(self.preview_table)
        preview.layout.addWidget(self.preview_table)
        page.layout().addWidget(preview, 1)
        return page

    def _build_analysis_page(self) -> QWidget:
        page = self._page()
        self._add_page_header(
            page,
            "Анализ текста",
            "Выберите файл, модель и параметры предобработки, затем запустите классификацию.",
        )

        mode_row = QHBoxLayout()
        mode_row.setSpacing(SPACE_1)
        self.analysis_mode_group = QButtonGroup(self)
        self.analysis_mode_group.setExclusive(True)
        batch_button = QPushButton("Пакетный анализ")
        batch_button.setObjectName("modeButton")
        batch_button.setCheckable(True)
        batch_button.setChecked(True)
        quick_button = QPushButton("Проверка текста")
        quick_button.setObjectName("modeButton")
        quick_button.setCheckable(True)
        for index, button in enumerate((batch_button, quick_button)):
            self.analysis_mode_group.addButton(button, index)
            button.clicked.connect(lambda checked=False, page_index=index: self._switch_analysis_mode(page_index))
            mode_row.addWidget(button)
        mode_row.addStretch(1)
        self.analysis_mode_hint = QLabel("Пакетный анализ использует отдельный файл анализа.")
        self.analysis_mode_hint.setObjectName("mutedLabel")
        mode_row.addWidget(self.analysis_mode_hint)
        page.layout().addLayout(mode_row)

        settings = Panel("Настройка и запуск анализа")
        settings_form = QVBoxLayout()
        settings_form.setContentsMargins(0, 0, 0, 0)
        settings_form.setSpacing(SPACE_2)

        self.analysis_data_row_widget = QWidget()
        analysis_data_row = QHBoxLayout(self.analysis_data_row_widget)
        analysis_data_row.setContentsMargins(0, 0, 0, 0)
        analysis_data_row.setSpacing(SPACE_2)
        self.analysis_load_button = QPushButton("Загрузить файл анализа")
        self.analysis_load_button.setObjectName("primaryButton")
        self.analysis_load_button.clicked.connect(self.load_analysis_dataset)
        self.analysis_clear_button = QPushButton("Очистить")
        self.analysis_clear_button.clicked.connect(self.clear_analysis_dataset)
        self.analysis_dataset_label = QLabel("Файл для пакетного анализа не загружен.")
        self.analysis_dataset_label.setObjectName("mutedLabel")
        self.analysis_dataset_label.setWordWrap(True)
        self.analysis_file_label = QLabel("Файл:")
        analysis_data_row.addWidget(self.analysis_file_label)
        analysis_data_row.addWidget(self.analysis_load_button)
        analysis_data_row.addWidget(self.analysis_clear_button)
        analysis_data_row.addWidget(self.analysis_dataset_label, 1)
        settings_form.addWidget(self.analysis_data_row_widget)
        self.analysis_file_separator = self._soft_separator()
        settings_form.addWidget(self.analysis_file_separator)

        self.text_column_combo = QComboBox()
        self.text_column_combo.addItem("text")
        self.text_column_combo.setFixedWidth(176)
        self.text_column_label = QLabel("Текст:")

        self.model_combo = QComboBox()
        for model_path in self._inference_local_models():
            if self.model_combo.findText(model_path) == -1:
                self.model_combo.addItem(model_path)
        self.model_combo.setEditable(True)
        self.model_combo.setMinimumWidth(360)
        self.model_combo.currentTextChanged.connect(self.populate_profile_table)

        self.lowercase_check = QCheckBox("Нижний регистр")
        self.punctuation_check = QCheckBox("Очистка пунктуации")
        self.stop_words_check = QCheckBox("Стоп-слова")
        self.lemmatize_check = QCheckBox("Лемматизация")
        for checkbox in (self.lowercase_check, self.punctuation_check, self.stop_words_check, self.lemmatize_check):
            checkbox.setChecked(False)

        self.analyze_button = QPushButton("Запустить анализ")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.setMinimumWidth(168)
        self.analyze_button.clicked.connect(self.run_analysis)

        self.browse_model_button = QPushButton("Выбрать папку…")
        self.browse_model_button.setToolTip(
            "Указать локальную папку с обученной моделью (config.json, model.safetensors, "
            "tokenizer_config.json, special_tokens_map.json и т. п.)."
        )
        self.browse_model_button.clicked.connect(self.browse_custom_model)

        model_row = QHBoxLayout()
        model_row.setSpacing(SPACE_2)
        model_row.addWidget(self.text_column_label)
        model_row.addWidget(self.text_column_combo)
        self.model_label = QLabel("Модель:")
        model_row.addWidget(self.model_label)
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.browse_model_button)
        settings_form.addLayout(model_row)
        settings_form.addWidget(self._soft_separator())

        preprocess_row = QHBoxLayout()
        preprocess_row.setSpacing(SPACE_3)
        preprocess_row.addWidget(QLabel("Предобработка:"))
        preprocess_row.addWidget(self.lowercase_check)
        preprocess_row.addWidget(self.punctuation_check)
        preprocess_row.addWidget(self.stop_words_check)
        preprocess_row.addWidget(self.lemmatize_check)
        preprocess_row.addStretch(1)
        preprocess_row.addWidget(self.analyze_button)
        settings_form.addLayout(preprocess_row)

        settings.layout.addLayout(settings_form)
        hint = QLabel("Для transformer-моделей агрессивную предобработку обычно лучше оставлять выключенной, если модель обучалась на сыром тексте.")
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        hint.setContentsMargins(0, 0, 0, 0)
        hint.setMaximumHeight(24)
        settings.layout.addWidget(hint)
        page.layout().addWidget(settings)

        self.analysis_mode_stack = QStackedWidget()
        batch_page = QWidget()
        batch_layout = QVBoxLayout(batch_page)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.setSpacing(SPACE_2)

        self.analysis_metrics_strip = QFrame()
        self.analysis_metrics_strip.setObjectName("analysisMetricsStrip")
        metrics = QHBoxLayout(self.analysis_metrics_strip)
        metrics.setContentsMargins(SPACE_3, 0, SPACE_3, 0)
        metrics.setSpacing(SPACE_4)
        self.total_metric_label = self._analysis_metric_label("Всего", "0")
        self.processed_metric_label = self._analysis_metric_label("Проанализировано", "0")
        self.confidence_metric_label = self._analysis_metric_label("Средняя уверенность", "0.00")
        self.low_confidence_metric_label = self._analysis_metric_label("На разметку", "0")
        for label in (
            self.total_metric_label,
            self.processed_metric_label,
            self.confidence_metric_label,
            self.low_confidence_metric_label,
        ):
            metrics.addWidget(label)
            metrics.addStretch(1)
        batch_layout.addWidget(self.analysis_metrics_strip)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_results_panel())
        splitter.addWidget(self._build_analysis_right_panel())
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, True)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([980, 260])
        batch_layout.addWidget(splitter, 1)

        self.analysis_mode_stack.addWidget(batch_page)
        self.analysis_mode_stack.addWidget(self._build_quick_analysis_panel())
        page.layout().addWidget(self.analysis_mode_stack, 1)
        return page

    def _build_quick_analysis_panel(self) -> QWidget:
        panel = Panel()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        input_box = QWidget()
        input_layout = QVBoxLayout(input_box)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(SPACE_2)
        input_label = QLabel("Текст для проверки")
        input_label.setObjectName("formLabel")

        self.quick_text_edit = QPlainTextEdit()
        self.quick_text_edit.setPlaceholderText("Введите один русский текст для проверки без загрузки датасета")
        self.quick_text_edit.setObjectName("quickTextInput")
        self.quick_text_edit.setMinimumHeight(240)
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.quick_text_edit, 1)

        result_box = QFrame()
        result_box.setObjectName("quickResultPanel")
        result_box.setMinimumWidth(280)
        result_layout = QVBoxLayout(result_box)
        result_layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        result_layout.setSpacing(SPACE_3)

        self.quick_result_label = QLabel("Результат появится здесь")
        self.quick_result_label.setObjectName("quickResult")
        self.quick_result_label.setProperty("sentiment", "none")
        self.quick_result_label.setWordWrap(True)
        self.quick_probability_label = QLabel("Вероятности появятся после анализа")
        self.quick_probability_label.setObjectName("mutedLabel")
        self.quick_probability_label.setWordWrap(True)
        self.quick_confidence_bar = QProgressBar()
        self.quick_confidence_bar.setRange(0, 100)
        self.quick_confidence_bar.setValue(0)
        self.quick_confidence_bar.setFormat("Уверенность: %p%")

        self.quick_probability_table = QTableWidget(0, 2)
        self._configure_embedded_table(self.quick_probability_table)
        self.quick_probability_table.setObjectName("quickProbabilityTable")
        self.quick_probability_table.setHorizontalHeaderLabels(["Класс", "Вероятность"])
        self.quick_probability_table.setColumnWidth(0, 180)
        self.quick_probability_table.setColumnWidth(1, 92)
        self.quick_probability_table.setMaximumHeight(180)
        self._enable_fill_width_columns(self.quick_probability_table)

        button_row = QHBoxLayout()
        button_row.setSpacing(SPACE_2)
        self.quick_analyze_button = QPushButton("Запустить анализ")
        self.quick_analyze_button.setObjectName("primaryButton")
        self.quick_analyze_button.clicked.connect(self.run_quick_text_analysis)
        clear_button = QPushButton("Очистить")
        clear_button.clicked.connect(self.clear_quick_text_analysis)
        button_row.addWidget(self.quick_analyze_button, 1)
        button_row.addWidget(clear_button)

        result_layout.addWidget(self.quick_result_label)
        result_layout.addWidget(self.quick_probability_label)
        result_layout.addWidget(self.quick_confidence_bar)
        result_layout.addWidget(self.quick_probability_table)
        result_layout.addStretch(1)
        result_layout.addLayout(button_row)

        splitter.addWidget(input_box)
        splitter.addWidget(result_box)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([980, 280])

        panel.layout.addWidget(splitter, 1)
        return panel

    def _build_results_panel(self) -> QWidget:
        panel = Panel("Таблица результатов")
        header = QHBoxLayout()
        self.result_count_label = QLabel("Всего: 0")
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Фильтр по тексту или классу")
        self.filter_edit.setMinimumWidth(220)
        self.filter_edit.textChanged.connect(self.apply_filter)
        header.addWidget(self.result_count_label)
        header.addStretch(1)
        header.addWidget(self.filter_edit)
        panel.layout.addLayout(header)

        self.results_table = QTableWidget(0, 4)
        self._configure_embedded_table(self.results_table)
        self.results_table.setHorizontalHeaderLabels(["#", "Текст", "Класс", "Увер."])
        self.results_table.setColumnWidth(0, 45)
        self.results_table.setColumnWidth(1, 520)
        self.results_table.setColumnWidth(2, 125)
        self.results_table.setColumnWidth(3, 80)
        self._enable_fill_width_columns(self.results_table)
        panel.layout.addWidget(self.results_table, 1)
        return panel

    def _build_analysis_right_panel(self) -> QWidget:
        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_2)
        self.class_chart = ChartWidget("Сводка анализа: классы", 190)
        self.confidence_chart = ChartWidget("Сводка анализа: уверенность", 190)
        layout.addWidget(self.class_chart)
        layout.addWidget(self.confidence_chart)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(right)
        scroll.setMinimumWidth(300)
        return scroll

    def _build_training_page(self) -> QWidget:
        page = self._page()
        self._add_page_header(
            page,
            "Обучение модели",
            "Настройте эксперимент, метки и параметры дообучения; ход запуска отображается справа.",
        )
        splitter = QSplitter(Qt.Orientation.Horizontal)

        settings = Panel("Параметры обучения")
        settings.setObjectName("workbenchPanel")
        tabs = TrainingTabWidget()
        tabs.setObjectName("trainingTabs")
        tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.train_experiment_name_edit = QLineEdit("sentiment_experiment")
        self.train_description_edit = QPlainTextEdit()
        self.train_description_edit.setPlaceholderText("Краткое описание запуска, датасета или гипотезы")
        self.train_description_edit.setMaximumHeight(72)
        self.train_output_edit = QLineEdit("models/custom-sentiment")
        output_button = QPushButton("Выбрать")
        output_button.clicked.connect(self.select_training_output_dir)
        self.train_seed_spin = QSpinBox()
        self.train_seed_spin.setRange(0, 999999)
        self.train_seed_spin.setValue(42)
        tabs.addTab(
            self._training_tab([
                ("Название эксперимента", self.train_experiment_name_edit),
                ("Папка сохранения", self._row_widget(self.train_output_edit, output_button)),
                ("Seed", self.train_seed_spin),
                ("Описание", self.train_description_edit),
            ]),
            "Проект",
        )

        self.train_text_column_combo = QComboBox()
        self.train_text_column_combo.addItem("text")
        self.train_label_column_combo = QComboBox()
        self.train_label_column_combo.currentTextChanged.connect(self.refresh_label_mapping_table)
        self.train_drop_duplicates_check = QCheckBox("Удалять дубликаты")
        self.train_drop_duplicates_check.setChecked(True)
        self.train_shuffle_check = QCheckBox("Перемешивать данные")
        self.train_shuffle_check.setChecked(True)
        self.train_max_samples_spin = QSpinBox()
        self.train_max_samples_spin.setRange(0, 10_000_000)
        self.train_max_samples_spin.setValue(0)
        self.train_max_samples_spin.setSpecialValueText("без ограничения")
        tabs.addTab(
            self._training_tab([
                ("Колонка текста", self.train_text_column_combo),
                ("Колонка метки", self.train_label_column_combo),
                ("Максимум строк", self.train_max_samples_spin),
                ("Очистка", self._stack_widget(self.train_drop_duplicates_check, self.train_shuffle_check)),
            ]),
            "Данные",
        )

        tabs.addTab(self._build_label_mapping_tab(), "Метки")

        self.train_val_split_spin = QDoubleSpinBox()
        self.train_val_split_spin.setDecimals(2)
        self.train_val_split_spin.setRange(0.05, 0.5)
        self.train_val_split_spin.setSingleStep(0.05)
        self.train_val_split_spin.setValue(0.2)
        self.train_test_split_spin = QDoubleSpinBox()
        self.train_test_split_spin.setDecimals(2)
        self.train_test_split_spin.setRange(0.0, 0.5)
        self.train_test_split_spin.setSingleStep(0.05)
        self.train_test_split_spin.setValue(0.1)
        self.train_test_split_spin.setSpecialValueText("без test")
        self.train_stratify_check = QCheckBox("Стратифицировать по метке")
        self.train_stratify_check.setChecked(True)
        self.train_split_seed_spin = QSpinBox()
        self.train_split_seed_spin.setRange(0, 999999)
        self.train_split_seed_spin.setValue(42)
        self.train_class_weights_combo = QComboBox()
        self.train_class_weights_combo.addItems(["none", "balanced"])
        self.train_metric_combo = QComboBox()
        self.train_metric_combo.addItems(["macro_f1", "accuracy"])
        tabs.addTab(
            self._training_tab([
                ("Validation split", self.train_val_split_spin),
                ("Test split", self.train_test_split_spin),
                ("Split seed", self.train_split_seed_spin),
                ("Стратегия", self.train_stratify_check),
                ("Веса классов", self.train_class_weights_combo),
                ("Лучшая модель по", self.train_metric_combo),
            ]),
            "Выборка",
        )

        self.train_base_model_combo = QComboBox()
        self.train_base_model_combo.addItems(TRAINING_BASE_MODELS)
        self.train_base_model_combo.setEditable(True)
        for index, model_name in enumerate(TRAINING_BASE_MODELS):
            self.train_base_model_combo.setItemData(index, QColor("#1f5fa9"), Qt.ItemDataRole.ForegroundRole)
            self.train_base_model_combo.setItemData(index, QColor("#e8f0ff"), Qt.ItemDataRole.BackgroundRole)
            self.train_base_model_combo.setItemData(index, f"Hugging Face: {model_name}", Qt.ItemDataRole.ToolTipRole)
        self.train_local_only_check = QCheckBox("Только локальные файлы")
        self.train_local_only_check.setChecked(False)
        self.train_trust_remote_check = QCheckBox("trust_remote_code")
        self.train_ignore_mismatch_check = QCheckBox("Переинициализировать голову при несовпадении")
        self.train_ignore_mismatch_check.setChecked(True)
        self.train_problem_type_combo = QComboBox()
        self.train_problem_type_combo.addItems(["single_label_classification"])
        self.train_freeze_base_check = QCheckBox("Обучать только классификатор")
        self.train_freeze_embeddings_check = QCheckBox("Заморозить embeddings")
        self.train_freeze_layers_spin = QSpinBox()
        self.train_freeze_layers_spin.setRange(0, 48)
        self.train_freeze_layers_spin.setValue(0)
        tabs.addTab(
            self._training_tab([
                ("Базовая модель", self.train_base_model_combo),
                ("Тип задачи", self.train_problem_type_combo),
                ("Загрузка", self._stack_widget(self.train_local_only_check, self.train_trust_remote_check, self.train_ignore_mismatch_check)),
                ("Заморозка", self._stack_widget(self.train_freeze_base_check, self.train_freeze_embeddings_check)),
                ("Первые N слоёв", self.train_freeze_layers_spin),
            ]),
            "Модель",
        )

        self.train_max_length_spin = QSpinBox()
        self.train_max_length_spin.setRange(32, 512)
        self.train_max_length_spin.setValue(256)
        self.train_padding_combo = QComboBox()
        self.train_padding_combo.addItems(["max_length", "longest"])
        self.train_truncation_check = QCheckBox("Обрезать длинные тексты")
        self.train_truncation_check.setChecked(True)
        self.train_fast_tokenizer_check = QCheckBox("Fast tokenizer")
        self.train_fast_tokenizer_check.setChecked(True)
        self.train_pad_multiple_spin = QSpinBox()
        self.train_pad_multiple_spin.setRange(0, 64)
        self.train_pad_multiple_spin.setValue(8)
        self.train_pad_multiple_spin.setSpecialValueText("нет")
        tabs.addTab(
            self._training_tab([
                ("Max length", self.train_max_length_spin),
                ("Padding", self.train_padding_combo),
                ("Truncation", self._stack_widget(self.train_truncation_check)),
                ("Tokenizer", self._stack_widget(self.train_fast_tokenizer_check)),
                ("Pad to multiple of", self.train_pad_multiple_spin),
            ]),
            "Токены",
        )

        self.train_epochs_spin = QSpinBox()
        self.train_epochs_spin.setRange(1, 50)
        self.train_epochs_spin.setValue(3)
        self.train_batch_spin = QSpinBox()
        self.train_batch_spin.setRange(1, 128)
        self.train_batch_spin.setValue(8)
        self.train_eval_batch_spin = QSpinBox()
        self.train_eval_batch_spin.setRange(1, 256)
        self.train_eval_batch_spin.setValue(16)
        self.train_lr_spin = QDoubleSpinBox()
        self.train_lr_spin.setDecimals(6)
        self.train_lr_spin.setRange(0.000001, 0.01)
        self.train_lr_spin.setSingleStep(0.000001)
        self.train_lr_spin.setValue(0.00002)
        self.train_weight_decay_spin = QDoubleSpinBox()
        self.train_weight_decay_spin.setDecimals(4)
        self.train_weight_decay_spin.setRange(0.0, 1.0)
        self.train_weight_decay_spin.setSingleStep(0.01)
        self.train_weight_decay_spin.setValue(0.01)
        self.train_grad_accum_spin = QSpinBox()
        self.train_grad_accum_spin.setRange(1, 64)
        self.train_grad_accum_spin.setValue(1)
        self.train_max_grad_norm_spin = QDoubleSpinBox()
        self.train_max_grad_norm_spin.setDecimals(2)
        self.train_max_grad_norm_spin.setRange(0.0, 10.0)
        self.train_max_grad_norm_spin.setValue(1.0)
        tabs.addTab(
            self._training_tab([
                ("Эпохи", self.train_epochs_spin),
                ("Train batch", self.train_batch_spin),
                ("Eval batch", self.train_eval_batch_spin),
                ("Learning rate", self.train_lr_spin),
                ("Weight decay", self.train_weight_decay_spin),
                ("Gradient accumulation", self.train_grad_accum_spin),
                ("Gradient clipping", self.train_max_grad_norm_spin),
            ]),
            "Обучение",
        )

        self.train_optimizer_combo = QComboBox()
        self.train_optimizer_combo.addItems(["adamw_torch", "sgd"])
        self.train_beta1_spin = QDoubleSpinBox()
        self.train_beta1_spin.setDecimals(3)
        self.train_beta1_spin.setRange(0.0, 0.999)
        self.train_beta1_spin.setValue(0.9)
        self.train_beta2_spin = QDoubleSpinBox()
        self.train_beta2_spin.setDecimals(3)
        self.train_beta2_spin.setRange(0.0, 0.9999)
        self.train_beta2_spin.setValue(0.999)
        self.train_epsilon_spin = QDoubleSpinBox()
        self.train_epsilon_spin.setDecimals(10)
        self.train_epsilon_spin.setRange(0.0000000001, 0.001)
        self.train_epsilon_spin.setValue(0.00000001)
        self.train_scheduler_combo = QComboBox()
        self.train_scheduler_combo.addItems(["linear", "cosine", "constant"])
        self.train_warmup_ratio_spin = QDoubleSpinBox()
        self.train_warmup_ratio_spin.setDecimals(2)
        self.train_warmup_ratio_spin.setRange(0.0, 0.5)
        self.train_warmup_ratio_spin.setValue(0.1)
        self.train_warmup_steps_spin = QSpinBox()
        self.train_warmup_steps_spin.setRange(0, 1_000_000)
        self.train_label_smoothing_spin = QDoubleSpinBox()
        self.train_label_smoothing_spin.setDecimals(2)
        self.train_label_smoothing_spin.setRange(0.0, 0.5)
        tabs.addTab(
            self._training_tab([
                ("Оптимизатор", self.train_optimizer_combo),
                ("Adam beta1", self.train_beta1_spin),
                ("Adam beta2", self.train_beta2_spin),
                ("Adam epsilon", self.train_epsilon_spin),
                ("Scheduler", self.train_scheduler_combo),
                ("Warmup ratio", self.train_warmup_ratio_spin),
                ("Warmup steps", self.train_warmup_steps_spin),
                ("Label smoothing", self.train_label_smoothing_spin),
            ]),
            "Оптимизация",
        )

        self.train_early_stop_check = QCheckBox("Ранняя остановка")
        self.train_early_stop_check.setChecked(True)
        self.train_early_patience_spin = QSpinBox()
        self.train_early_patience_spin.setRange(1, 20)
        self.train_early_patience_spin.setValue(2)
        self.train_device_combo = QComboBox()
        self.train_device_combo.addItems(device_labels())
        self.train_fp16_check = QCheckBox("FP16 на CUDA")
        self.train_gradient_checkpointing_check = QCheckBox("Gradient checkpointing")
        self.train_workers_spin = QSpinBox()
        self.train_workers_spin.setRange(0, 16)
        self.train_pin_memory_check = QCheckBox("Pin memory")
        self.train_pin_memory_check.setChecked(True)
        self.train_save_predictions_check = QCheckBox("Сохранять predictions")
        self.train_save_predictions_check.setChecked(True)
        self.train_save_config_check = QCheckBox("Сохранять training_config.json")
        self.train_save_config_check.setChecked(True)
        tabs.addTab(
            self._training_tab([
                ("Early stopping", self._stack_widget(self.train_early_stop_check)),
                ("Patience", self.train_early_patience_spin),
                ("Устройство", self.train_device_combo),
                ("Precision", self._stack_widget(self.train_fp16_check, self.train_gradient_checkpointing_check)),
                ("Dataloader workers", self.train_workers_spin),
                ("Memory", self._stack_widget(self.train_pin_memory_check)),
                ("Сохранение", self._stack_widget(self.train_save_predictions_check, self.train_save_config_check)),
            ]),
            "Дополнительно",
        )

        for index, tooltip in enumerate([
            "Проект эксперимента",
            "Датасет и очистка",
            "Метки и исключения",
            "Разбиение выборки",
            "Базовая модель",
            "Токенизация",
            "Параметры обучения",
            "Оптимизация",
            "Дополнительные параметры",
        ]):
            tabs.setTabToolTip(index, tooltip)

        settings.layout.addWidget(tabs, 1)
        self.train_button = QPushButton("Запустить обучение")
        self.train_button.setObjectName("primaryButton")
        self.train_button.setMinimumWidth(168)
        self.train_button.clicked.connect(self.start_training)
        self.cancel_train_button = QPushButton("Прервать")
        self.cancel_train_button.setObjectName("dangerButton")
        self.cancel_train_button.clicked.connect(self.cancel_training)
        self.cancel_train_button.setEnabled(False)
        train_actions = QHBoxLayout()
        train_actions.setSpacing(SPACE_2)
        train_actions.addWidget(self.train_button, 1)
        train_actions.addWidget(self.cancel_train_button)
        settings.layout.addLayout(train_actions)
        settings.setMinimumWidth(420)
        settings.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        splitter.addWidget(settings)

        progress_panel = Panel("Ход выполнения")
        progress_panel.setObjectName("workbenchPanel")
        progress_panel.setMinimumWidth(260)
        progress_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 1)
        self.training_progress.setValue(0)
        self.training_metrics_chart = ChartWidget("Итоги обучения", 176)
        self.training_metrics_chart.empty("Метрики появятся после обучения")
        self.training_log = QPlainTextEdit()
        self.training_log.setObjectName("trainingLog")
        self.training_log.setReadOnly(True)
        self.training_log_stack = QStackedWidget()
        self.training_log_stack.setObjectName("trainingLogStack")
        log_empty = QFrame()
        log_empty.setObjectName("trainingLogEmpty")
        log_empty_layout = QVBoxLayout(log_empty)
        log_empty_layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        log_empty_layout.addStretch(1)
        log_empty_label = QLabel("Сообщения появятся при обучении")
        log_empty_label.setObjectName("emptyLogMessage")
        log_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        log_empty_layout.addWidget(log_empty_label)
        log_empty_layout.addStretch(1)
        self.training_log_stack.addWidget(log_empty)
        self.training_log_stack.addWidget(self.training_log)
        progress_panel.layout.addWidget(self.training_progress)
        progress_panel.layout.addWidget(self.training_metrics_chart)
        progress_panel.layout.addWidget(self.training_log_stack, 1)
        splitter.addWidget(progress_panel)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, True)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        splitter.setChildrenCollapsible(True)
        splitter.setSizes([620, 360])
        page.layout().addWidget(splitter, 1)
        return page

    def _build_label_mapping_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        layout.setSpacing(SPACE_2)

        form = QGridLayout()
        form.setHorizontalSpacing(SPACE_3)
        form.setVerticalSpacing(SPACE_3)
        form.setColumnMinimumWidth(0, 128)

        self.label_format_combo = QComboBox()
        self.label_format_combo.addItems(["Одна метка в строке", "Список меток в строке"])
        self.label_format_combo.currentTextChanged.connect(self.refresh_label_mapping_table)

        self.multilabel_strategy_combo = QComboBox()
        self.multilabel_strategy_combo.addItems([
            "Первая подходящая по таблице",
            "Исключать строки со списком меток",
            "Первая метка в списке",
        ])
        self.multilabel_strategy_combo.currentTextChanged.connect(self.update_label_mapping_summary)

        self.target_scheme_combo = QComboBox()
        self.target_scheme_combo.addItems([ORIGINAL_SCHEME, SENTIMENT_SCHEME])
        self.target_scheme_combo.currentTextChanged.connect(self.refresh_label_mapping_table)

        auto_button = QPushButton("Автонастроить метки")
        auto_button.clicked.connect(self.auto_configure_label_mapping)
        apply_button = QPushButton("Проверить разметку")
        apply_button.clicked.connect(self.update_label_mapping_summary)

        form.addWidget(self._field_label("Формат меток"), 0, 0)
        form.addWidget(self._field_cell("Формат меток", self.label_format_combo, show_help=False), 0, 1)
        form.addWidget(self._field_label("Если меток несколько"), 1, 0)
        form.addWidget(self._field_cell("Если меток несколько", self.multilabel_strategy_combo, show_help=False), 1, 1)
        form.addWidget(self._field_label("Схема обучения"), 2, 0)
        form.addWidget(self._field_cell("Схема обучения", self.target_scheme_combo, show_help=False), 2, 1)
        form.addWidget(auto_button, 3, 0)
        form.addWidget(apply_button, 3, 1)
        form.setColumnStretch(1, 1)
        layout.addLayout(form)

        self.label_mapping_table = QTableWidget(0, 4)
        self._configure_embedded_table(self.label_mapping_table)
        self.label_mapping_table.setHorizontalHeaderLabels(["Метка", "Кол-во", "Действие", "Новая метка"])
        self.label_mapping_table.setColumnWidth(0, 220)
        self.label_mapping_table.setColumnWidth(1, 72)
        self.label_mapping_table.setColumnWidth(2, 220)
        self.label_mapping_table.setColumnWidth(3, 220)
        self._enable_fill_width_columns(self.label_mapping_table)

        self.label_mapping_table.verticalHeader().setDefaultSectionSize(44)
        self.label_mapping_table.verticalHeader().setMinimumSectionSize(44)
        layout.addWidget(self.label_mapping_table, 1)

        self.label_mapping_summary = QLabel("Загрузите датасет и выберите колонку метки.")
        self.label_mapping_summary.setObjectName("mappingSummary")
        self.label_mapping_summary.setWordWrap(True)
        self.label_mapping_summary.hide()
        manual_hint = QLabel(
            "В столбце новой метки можно вручную задать целевое имя класса для любого исходного значения, "
            "например 0 -> нейтральная или 1 -> положительная."
        )
        manual_hint.setObjectName("mutedLabel")
        manual_hint.setWordWrap(True)
        layout.addWidget(manual_hint)
        return content

    def _build_models_page(self) -> QWidget:
        page = self._page()
        self._add_page_header(
            page,
            "Модели",
            "Просматривайте локальные обученные модели, их метки и сохранённые метрики качества.",
        )
        panel = Panel("Реестр локальных моделей")
        note = QLabel(
            "Реестр показывает локальные обученные модели, их схему меток и пригодность для сравнения качества. "
            "Базовые контрольные точки нужны только для дообучения и не добавляются в список анализа."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        panel.layout.addWidget(note)
        self.profile_table = QTableWidget(0, 7)
        self._configure_embedded_table(self.profile_table)
        self.profile_table.setHorizontalHeaderLabels(["Модель", "Источник", "Схема меток", "Качество", "Accuracy", "Macro F1", "Действия"])
        self.profile_table.setColumnWidth(0, 200)
        self.profile_table.setColumnWidth(1, 132)
        self.profile_table.setColumnWidth(2, 240)
        self.profile_table.setColumnWidth(3, 122)
        self.profile_table.setColumnWidth(4, 88)
        self.profile_table.setColumnWidth(5, 88)
        self.profile_table.setColumnWidth(6, 170)
        self._enable_fill_width_columns(self.profile_table)
        self.profile_table.verticalHeader().setDefaultSectionSize(self._model_action_row_height())
        self.profile_table.verticalHeader().setMinimumSectionSize(self._model_action_row_height())
        self.profile_table.cellDoubleClicked.connect(self._select_model_from_profile)
        self.profile_table.itemChanged.connect(self._handle_profile_item_changed)
        panel.layout.addWidget(self.profile_table, 1)
        page.layout().addWidget(panel, 1)
        return page

    def _build_comparison_page(self) -> QWidget:
        page = self._page()
        self._add_page_header(
            page,
            "Сравнение моделей",
            "Сопоставьте несколько моделей на одном наборе текстов и проверьте расхождения в ответах.",
        )
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        controls = Panel("Параметры сравнения")
        intro = QLabel(
            "Качество считается только для моделей с совпадающей схемой меток и при выбранной колонке истинной метки. "
            "Поведение можно сравнивать между любыми моделями."
        )
        intro.setWordWrap(True)
        intro.setObjectName("mutedLabel")
        controls.layout.addWidget(intro)

        toolbar = QFrame()
        toolbar.setObjectName("comparisonToolbar")
        toolbar_layout = QVBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        toolbar_layout.setSpacing(SPACE_2)

        data_grid = QGridLayout()
        data_grid.setHorizontalSpacing(SPACE_3)
        data_grid.setVerticalSpacing(SPACE_1)
        data_grid.addWidget(self._comparison_field_label("Источник данных"), 0, 0)
        data_grid.addWidget(self._comparison_field_label("Текст"), 0, 1)
        data_grid.addWidget(self._comparison_field_label("Метка"), 0, 2)
        self.comparison_dataset_combo = QComboBox()
        self.comparison_dataset_combo.setMinimumWidth(180)
        self.comparison_dataset_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.comparison_dataset_combo.currentTextChanged.connect(self._populate_comparison_columns)
        data_grid.addWidget(self.comparison_dataset_combo, 1, 0)
        self.comparison_text_column_combo = QComboBox()
        self.comparison_text_column_combo.setMinimumWidth(180)
        self.comparison_text_column_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        data_grid.addWidget(self.comparison_text_column_combo, 1, 1)
        self.comparison_label_column_combo = QComboBox()
        self.comparison_label_column_combo.setMinimumWidth(180)
        self.comparison_label_column_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        data_grid.addWidget(self.comparison_label_column_combo, 1, 2)
        data_grid.setColumnStretch(0, 1)
        data_grid.setColumnStretch(1, 1)
        data_grid.setColumnStretch(2, 1)
        toolbar_layout.addLayout(data_grid)

        sample_grid = QGridLayout()
        sample_grid.setHorizontalSpacing(SPACE_3)
        sample_grid.setVerticalSpacing(SPACE_1)
        sample_grid.addWidget(self._comparison_field_label("Отбор строк"), 0, 0)
        sample_grid.addWidget(self._comparison_field_label("Максимум строк"), 0, 1)
        self.comparison_sampling_combo = QComboBox()
        self.comparison_sampling_combo.addItem("По порядку", "ordered")
        self.comparison_sampling_combo.addItem("Случайно", "random")
        self.comparison_sampling_combo.setMinimumWidth(140)
        sample_grid.addWidget(self.comparison_sampling_combo, 1, 0)
        self.comparison_max_rows_spin = QSpinBox()
        self.comparison_max_rows_spin.setRange(0, 10_000_000)
        self.comparison_max_rows_spin.setValue(0)
        self.comparison_max_rows_spin.setSpecialValueText("все строки")
        self.comparison_max_rows_spin.setMinimumWidth(140)
        sample_grid.addWidget(self.comparison_max_rows_spin, 1, 1)
        sample_grid.setColumnStretch(0, 1)
        sample_grid.setColumnStretch(1, 1)
        toolbar_layout.addLayout(sample_grid)

        sampling_note = QLabel(
            "0 = весь датасет. При ограничении можно взять первые N строк или случайную подвыборку."
        )
        sampling_note.setObjectName("comparisonHint")
        sampling_note.setWordWrap(True)
        toolbar_layout.addWidget(sampling_note)
        controls.layout.addWidget(toolbar)

        self.comparison_status_label = QLabel("Выберите минимум две модели и датасет для сравнения.")
        self.comparison_status_label.setObjectName("comparisonStatusBox")
        self.comparison_status_label.setWordWrap(True)
        controls.layout.addWidget(self.comparison_status_label)

        self.comparison_model_table = QTableWidget(0, 4)
        self._configure_embedded_table(self.comparison_model_table)
        self.comparison_model_table.setHorizontalHeaderLabels(["Сравнить", "Модель", "Источник", "Схема меток"])
        self.comparison_model_table.setColumnWidth(0, 82)
        self.comparison_model_table.setColumnWidth(1, 220)
        self.comparison_model_table.setColumnWidth(2, 132)
        self.comparison_model_table.setColumnWidth(3, 220)
        self._enable_fill_width_columns(self.comparison_model_table)
        controls.layout.addWidget(self.comparison_model_table, 1)

        compare_row = QHBoxLayout()
        compare_row.setSpacing(SPACE_2)
        self.run_comparison_button = QPushButton("Сравнить модели")
        self.run_comparison_button.setObjectName("primaryButton")
        self.run_comparison_button.clicked.connect(self.run_model_comparison)
        select_all_button = QPushButton("Выбрать все")
        select_all_button.clicked.connect(lambda: self._set_comparison_selection(True))
        clear_all_button = QPushButton("Снять выбор")
        clear_all_button.clicked.connect(lambda: self._set_comparison_selection(False))
        compare_row.addWidget(self.run_comparison_button)
        compare_row.addWidget(select_all_button)
        compare_row.addWidget(clear_all_button)
        compare_row.addStretch(1)
        controls.layout.addLayout(compare_row)

        results = QWidget()
        results.setMinimumWidth(460)
        results_layout = QVBoxLayout(results)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(SPACE_3)

        results_tabs = TrainingTabWidget()
        results_tabs.setObjectName("comparisonTabs")

        charts_content = QWidget()
        charts_layout = QVBoxLayout(charts_content)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(SPACE_3)
        self.comparison_quality_chart = ChartWidget("Качество по моделям", 260)
        self.comparison_behavior_chart = ChartWidget("Уверенность и сомнения", 260)
        self.comparison_speed_chart = ChartWidget("Время анализа", 240)
        self.comparison_disagreement_chart = ChartWidget("Расхождения между моделями", 280)
        self.comparison_quality_chart.empty("График качества появится после сравнения")
        self.comparison_behavior_chart.empty("График поведения появится после сравнения")
        self.comparison_speed_chart.empty("График времени появится после сравнения")
        self.comparison_disagreement_chart.empty("График расхождений появится после сравнения")
        charts_layout.addWidget(self.comparison_quality_chart)
        charts_layout.addWidget(self.comparison_behavior_chart)
        charts_layout.addWidget(self.comparison_speed_chart)
        charts_layout.addWidget(self.comparison_disagreement_chart)
        charts_layout.addStretch(1)

        charts_scroll = QScrollArea()
        charts_scroll.setWidgetResizable(True)
        charts_scroll.setFrameShape(QFrame.Shape.NoFrame)
        charts_scroll.setWidget(charts_content)
        results_tabs.addTab(charts_scroll, "Графики")

        tables_page = QWidget()
        tables_layout = QVBoxLayout(tables_page)
        tables_layout.setContentsMargins(0, 0, 0, 0)
        tables_layout.setSpacing(SPACE_3)

        quality_panel = Panel("Качество")
        self.comparison_quality_summary = QLabel("Сравнение качества появится после запуска.")
        self.comparison_quality_summary.setObjectName("mutedLabel")
        self.comparison_quality_summary.setWordWrap(True)
        quality_panel.layout.addWidget(self.comparison_quality_summary)
        self.comparison_quality_table = QTableWidget(0, 5)
        self._configure_embedded_table(self.comparison_quality_table)
        self.comparison_quality_table.setHorizontalHeaderLabels(["Модель", "Метки", "Строк", "Accuracy", "Macro F1"])
        self.comparison_quality_table.setColumnWidth(0, 180)
        self.comparison_quality_table.setColumnWidth(1, 260)
        self.comparison_quality_table.setColumnWidth(2, 84)
        self.comparison_quality_table.setColumnWidth(3, 90)
        self.comparison_quality_table.setColumnWidth(4, 90)
        self._enable_fill_width_columns(self.comparison_quality_table)
        quality_panel.layout.addWidget(self.comparison_quality_table, 1)
        tables_layout.addWidget(quality_panel, 1)

        behavior_panel = Panel("Поведение")
        self.comparison_behavior_table = QTableWidget(0, 8)
        self._configure_embedded_table(self.comparison_behavior_table)
        self.comparison_behavior_table.setHorizontalHeaderLabels(
            ["Модель", "Метки", "Средняя уверенность", "Медиана", "Низкая уверенность", "Энтропия", "Доля топ-класса", "Секунды"]
        )
        self.comparison_behavior_table.setColumnWidth(0, 180)
        self.comparison_behavior_table.setColumnWidth(1, 260)
        self.comparison_behavior_table.setColumnWidth(2, 150)
        self.comparison_behavior_table.setColumnWidth(3, 96)
        self.comparison_behavior_table.setColumnWidth(4, 150)
        self.comparison_behavior_table.setColumnWidth(5, 96)
        self.comparison_behavior_table.setColumnWidth(6, 132)
        self.comparison_behavior_table.setColumnWidth(7, 96)
        self._enable_fill_width_columns(self.comparison_behavior_table)
        behavior_panel.layout.addWidget(self.comparison_behavior_table, 1)
        tables_layout.addWidget(behavior_panel, 1)

        disagreement_panel = Panel("Расхождения")
        self.comparison_disagreement_table = QTableWidget(0, 3)
        self._configure_embedded_table(self.comparison_disagreement_table)
        self.comparison_disagreement_table.setHorizontalHeaderLabels(["#", "Текст", "Предсказания"])
        self.comparison_disagreement_table.setColumnWidth(0, 44)
        self.comparison_disagreement_table.setColumnWidth(1, 320)
        self.comparison_disagreement_table.setColumnWidth(2, 420)
        self._enable_fill_width_columns(self.comparison_disagreement_table)
        disagreement_panel.layout.addWidget(self.comparison_disagreement_table, 1)
        tables_layout.addWidget(disagreement_panel, 1)

        results_tabs.addTab(tables_page, "Таблицы")
        results_layout.addWidget(results_tabs, 1)

        splitter.addWidget(controls)
        splitter.addWidget(results)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([440, 820])

        page.layout().addWidget(splitter, 1)
        return page

    def _build_monitoring_page(self) -> QWidget:
        page = self._page()
        self._add_page_header(
            page,
            "Мониторинг результатов",
            "После анализа здесь отображаются уверенность, распределения классов и признаки изменения данных.",
        )
        self.monitoring_stack = QStackedWidget()

        self.monitoring_stack.addWidget(
            self._empty_state_panel(
                title="Мониторинг появится после анализа",
                message="Запустите пакетный анализ, чтобы здесь появились динамика уверенности, распределение классов и сводка дрейфа.",
                action_label="Перейти к анализу",
                action_callback=lambda: self._switch_page(1),
                secondary_label="К данным",
                secondary_callback=lambda: self._switch_page(0),
            )
        )

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(SPACE_3)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.drift_chart = ChartWidget("Динамика уверенности и распределений", 320)
        splitter.addWidget(self.drift_chart)
        panel = Panel("Показатели мониторинга")
        self.monitoring_summary_label = QLabel("Недостаточно данных для мониторинга.")
        self.monitoring_summary_label.setWordWrap(True)
        self.monitoring_summary_label.setObjectName("mutedLabel")
        self.monitoring_table = QTableWidget(0, 2)
        self._configure_embedded_table(self.monitoring_table)
        self.monitoring_table.setHorizontalHeaderLabels(["Показатель", "Значение"])
        self.monitoring_table.setColumnWidth(0, 220)
        self.monitoring_table.setColumnWidth(1, 120)
        self._enable_fill_width_columns(self.monitoring_table)
        panel.layout.addWidget(self.monitoring_summary_label)
        panel.layout.addWidget(self.monitoring_table, 1)
        splitter.addWidget(panel)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, True)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([760, 320])
        content_layout.addWidget(splitter, 1)
        self.monitoring_stack.addWidget(content)

        page.layout().addWidget(self.monitoring_stack, 1)
        return page

    def _build_reports_page(self) -> QWidget:
        page = self._page()
        self._add_page_header(
            page,
            "Отчёты",
            "Сформируйте HTML-отчёт для демонстрации результатов анализа и качества модели.",
        )
        self.reports_stack = QStackedWidget()

        self.reports_stack.addWidget(
            self._empty_state_panel(
                title="Отчёт сформируется после анализа",
                message="После запуска пакетного анализа здесь можно будет экспортировать HTML-отчёт с результатами и метриками.",
                action_label="Перейти к анализу",
                action_callback=lambda: self._switch_page(1),
                secondary_label="К данным",
                secondary_callback=lambda: self._switch_page(0),
            )
        )

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(SPACE_3)
        panel = Panel("Экспорт и предпросмотр HTML-отчёта")
        toolbar = QHBoxLayout()
        toolbar.setSpacing(SPACE_2)
        export_button = QPushButton("Сохранить HTML")
        export_button.clicked.connect(self.export_report)
        toolbar.addWidget(export_button)
        toolbar.addStretch(1)

        self.report_view = QTextBrowser()
        self.report_view.setObjectName("reportPreview")
        self.report_view.setOpenExternalLinks(True)
        self.report_view.setHtml(
            "<div style='height:100%; display:flex; align-items:center; justify-content:center; "
            "font-family:Segoe UI, Arial; color:#6b7280; font-size:14px;'>"
            "Предпросмотр появится после формирования отчёта"
            "</div>"
        )

        panel.layout.addLayout(toolbar)
        panel.layout.addWidget(self.report_view, 1)
        content_layout.addWidget(panel)
        self.reports_stack.addWidget(content)

        page.layout().addWidget(self.reports_stack, 1)
        return page

    def _empty_state_panel(
        self,
        title: str,
        message: str,
        action_label: str | None = None,
        action_callback=None,
        *,
        kicker: str | None = None,
        steps: list[str] | None = None,
        secondary_label: str | None = None,
        secondary_callback=None,
    ) -> QWidget:
        wrapper = QWidget()
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(SPACE_4, SPACE_4, SPACE_4, SPACE_4)
        outer.setSpacing(SPACE_2)
        outer.addStretch(1)

        if kicker:
            kicker_label = QLabel(kicker.upper())
            kicker_label.setObjectName("emptyStateKicker")
            kicker_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            outer.addWidget(kicker_label)

        if title:
            title_label = QLabel(title)
            title_label.setObjectName("emptyStateTitle")
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setWordWrap(True)
            outer.addWidget(title_label)

        message_label = QLabel(message)
        message_label.setObjectName("emptyStateMessage")
        message_label.setWordWrap(True)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setMaximumWidth(520)
        message_container = QHBoxLayout()
        message_container.addStretch(1)
        message_container.addWidget(message_label)
        message_container.addStretch(1)
        outer.addLayout(message_container)

        if steps:
            steps_widget = QWidget()
            steps_layout = QVBoxLayout(steps_widget)
            steps_layout.setContentsMargins(0, SPACE_2, 0, SPACE_2)
            steps_layout.setSpacing(SPACE_1)
            for index, step in enumerate(steps, start=1):
                step_label = QLabel(f"{index}.  {step}")
                step_label.setObjectName("emptyStateStepText")
                step_label.setWordWrap(True)
                step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                steps_layout.addWidget(step_label)
            steps_widget.setMaximumWidth(520)
            steps_row = QHBoxLayout()
            steps_row.addStretch(1)
            steps_row.addWidget(steps_widget)
            steps_row.addStretch(1)
            outer.addLayout(steps_row)

        if action_label and action_callback is not None:
            button_row = QHBoxLayout()
            button_row.setSpacing(SPACE_2)
            button_row.addStretch(1)
            action_button = QPushButton(action_label)
            action_button.setObjectName("primaryButton")
            action_button.clicked.connect(action_callback)
            button_row.addWidget(action_button)
            if secondary_label and secondary_callback is not None:
                secondary_button = QPushButton(secondary_label)
                secondary_button.clicked.connect(secondary_callback)
                button_row.addWidget(secondary_button)
            button_row.addStretch(1)
            outer.addSpacing(SPACE_2)
            outer.addLayout(button_row)

        outer.addStretch(1)
        return wrapper

    def _training_tab(self, rows: list[tuple[str, QWidget]]) -> QWidget:
        content = QWidget()
        form = QGridLayout(content)
        form.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        form.setHorizontalSpacing(SPACE_3)
        form.setVerticalSpacing(SPACE_2)
        for row, (label, widget) in enumerate(rows):
            form.addWidget(self._field_label(label), row, 0)
            form.addWidget(self._field_cell(label, widget), row, 1)
        form.setColumnStretch(1, 1)
        form.setRowStretch(len(rows), 1)

        scroll = QScrollArea()
        scroll.setObjectName("trainingScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _field_label(self, label: str) -> QLabel:
        label_widget = QLabel(label)
        label_widget.setObjectName("formLabel")
        label_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        help_text = TRAINING_FIELD_HELP.get(label, "")
        if help_text:
            label_widget.setToolTip(help_text)
        return label_widget

    def _analysis_metric_label(self, title: str, value: str) -> QLabel:
        label = QLabel(f"{title}: {value}")
        label.setObjectName("analysisMetricText")
        return label

    def _comparison_field_label(self, label: str) -> QLabel:
        label_widget = QLabel(label)
        label_widget.setObjectName("comparisonFieldLabel")
        return label_widget

    def _field_cell(self, label: str, widget: QWidget, show_help: bool = True) -> QWidget:
        help_text = TRAINING_FIELD_HELP.get(label, "")
        if help_text:
            self._apply_help_tooltip(widget, help_text)

        if not help_text or not show_help:
            return widget

        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_1)
        layout.addWidget(widget)
        help_label = QLabel(help_text)
        help_label.setObjectName("fieldHelp")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        return wrapper

    def _apply_help_tooltip(self, widget: QWidget, help_text: str) -> None:
        widget.setToolTip(help_text)
        for child in widget.findChildren(QWidget):
            child.setToolTip(help_text)

    def _row_widget(self, *widgets: QWidget) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_2)
        for index, widget in enumerate(widgets):
            layout.addWidget(widget, 1 if index == 0 else 0)
        return wrapper

    def _stack_widget(self, *widgets: QWidget) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_1)
        for widget in widgets:
            layout.addWidget(widget)
        return wrapper

    def _soft_separator(self) -> QFrame:
        separator = QFrame()
        separator.setObjectName("softSeparator")
        separator.setFrameShape(QFrame.Shape.NoFrame)
        separator.setFixedHeight(1)
        return separator

    def _add_page_header(self, page: QWidget, title: str, subtitle: str) -> None:
        header = QFrame()
        header.setObjectName("pageHeader")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(SPACE_3, SPACE_2, SPACE_3, SPACE_2)
        layout.setSpacing(SPACE_1)

        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("pageSubtitle")
        subtitle_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        page.layout().addWidget(header)

    def _page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        layout.setSpacing(SPACE_3)
        return page

    def _configure_embedded_table(self, table: QTableWidget) -> None:
        table.setObjectName("embeddedTable")
        table.setAlternatingRowColors(True)
        table.setShowGrid(True)
        table.setWordWrap(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        header = table.horizontalHeader()
        header.setHighlightSections(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setCascadingSectionResizes(False)
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(48)
        table.verticalHeader().setVisible(False)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _enable_fill_width_columns(self, table: QTableWidget) -> None:
        table_id = id(table)
        viewport_id = id(table.viewport())
        self._fill_width_tables[table_id] = table
        self._fill_width_viewports[viewport_id] = table
        table.viewport().installEventFilter(self)
        table.installEventFilter(self)
        table.horizontalHeader().sectionResized.connect(
            lambda section, old_size, new_size, target=table: self._rebalance_table_columns(target, section)
        )
        self._rebalance_table_columns(table)

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        table = self._fill_width_viewports.get(id(watched))
        if table is None and isinstance(watched, QTableWidget):
            table = self._fill_width_tables.get(id(watched))
        if table is not None and event.type() in {QEvent.Type.Resize, QEvent.Type.Show}:
            self._rebalance_table_columns(table)
        return super().eventFilter(watched, event)

    def _rebalance_table_columns(self, table: QTableWidget, preferred_section: int | None = None) -> None:
        if table.columnCount() == 0:
            return
        table_id = id(table)
        if table_id in self._table_resize_guard:
            return

        viewport_width = max(table.viewport().width(), 0)
        if viewport_width <= 0:
            return

        header = table.horizontalHeader()
        widths = [header.sectionSize(index) for index in range(table.columnCount())]
        total_width = sum(widths)
        min_width = max(header.minimumSectionSize(), 48)
        target_width = max(viewport_width - 2, min_width * table.columnCount())
        delta = target_width - total_width
        if delta == 0:
            return

        self._table_resize_guard.add(table_id)
        try:
            if delta > 0:
                last_index = table.columnCount() - 1
                table.setColumnWidth(last_index, widths[last_index] + delta)
                return

            remaining = -delta
            candidate_indexes = [
                index
                for index in range(table.columnCount() - 1, -1, -1)
                if index != preferred_section
            ]
            if preferred_section is not None:
                candidate_indexes.append(preferred_section)

            current_widths = widths[:]
            for index in candidate_indexes:
                reducible = max(current_widths[index] - min_width, 0)
                if reducible <= 0:
                    continue
                shrink = min(reducible, remaining)
                if shrink > 0:
                    current_widths[index] -= shrink
                    remaining -= shrink
                if remaining <= 0:
                    break

            for index, width in enumerate(current_widths):
                table.setColumnWidth(index, width)
        finally:
            self._table_resize_guard.discard(table_id)

    def _set_status(self, text: str, state: str = "ready") -> None:
        if hasattr(self, "status_label"):
            self.status_label.setText(f"● {text}")
            self.status_label.setProperty("state", state)
            self.status_label.style().unpolish(self.status_label)
            self.status_label.style().polish(self.status_label)
        bar = self.statusBar()
        bar.setProperty("state", state)
        bar.style().unpolish(bar)
        bar.style().polish(bar)
        bar.showMessage(text)

    def _set_initial_state(self) -> None:
        self.analyze_button.setEnabled(False)
        self.analysis_clear_button.setEnabled(False)
        self.train_button.setEnabled(False)
        self._switch_page(0)
        self.populate_profile_table()
        self.populate_comparison_model_table()
        self._refresh_comparison_inputs()
        self._refresh_empty_charts()
        self._refresh_workflow_state()
        self._refresh_dataset_summary()
        self._update_context_bar()
        self._log("Готово. Загрузите датасет для начала работы.")
        self.refresh_label_mapping_table()

    def _refresh_workflow_state(self) -> None:
        has_results = bool(self.results)
        if hasattr(self, "monitoring_stack"):
            self.monitoring_stack.setCurrentIndex(1 if has_results else 0)
        if hasattr(self, "reports_stack"):
            self.reports_stack.setCurrentIndex(1 if has_results else 0)

    def _switch_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        if 0 <= index < len(self.nav_buttons):
            self.nav_buttons[index].setChecked(True)

    def _switch_analysis_mode(self, index: int) -> None:
        self.analysis_mode_stack.setCurrentIndex(index)
        is_batch = index == 0
        self.analyze_button.setVisible(is_batch)
        self.text_column_combo.setVisible(is_batch)
        self.text_column_label.setVisible(is_batch)
        self.analysis_data_row_widget.setVisible(is_batch)
        self.analysis_file_separator.setVisible(is_batch)
        if hasattr(self, "analysis_mode_hint"):
            if is_batch:
                self.analysis_mode_hint.setText("Пакетный анализ использует отдельный файл анализа, не train-датасет.")
            else:
                self.analysis_mode_hint.setText("Быстрая проверка одной строки без загрузки датасета.")

    def load_train_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить train-датасет",
            str(Path.cwd()),
            "Текстовые данные (*.csv *.txt *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx *.xls);;TXT (*.txt)",
        )
        if not path:
            return

        try:
            self.data_frame = self._read_dataset(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))
            return

        self.dataset_path = Path(path)
        self._populate_training_columns()
        self._refresh_comparison_inputs()
        self.results = []
        self.preview_offset = 0
        self._refresh_dataset_summary()
        self.populate_results_table([])
        self.refresh_analysis()
        self.refresh_label_mapping_table()
        self.train_button.setEnabled(True)
        self._set_status("Данные загружены", "ready")
        self.statusBar().showMessage(
            "Train загружен. Можно загрузить отдельные validation/test или оставить разбиение по проценту."
        )
        self._log(f"Train загружен: {self.dataset_path.name} ({len(self.data_frame)} строк).")
        self._refresh_workflow_state()
        self._switch_page(0)

    def clear_train_dataset(self) -> None:
        if self.data_frame.empty:
            return
        self.data_frame = pd.DataFrame()
        self.dataset_path = None
        self.results = []
        self.preview_offset = 0
        self.train_text_column_combo.clear()
        self.train_text_column_combo.addItem("text")
        self.train_label_column_combo.blockSignals(True)
        self.train_label_column_combo.clear()
        self.train_label_column_combo.blockSignals(False)
        self.train_button.setEnabled(False)
        self._refresh_comparison_inputs()
        self._refresh_dataset_summary()
        self.populate_results_table([])
        self.refresh_analysis()
        self.refresh_label_mapping_table()
        self._set_status("Train очищен", "ready")
        self._log("Train-датасет очищен.")
        self._refresh_workflow_state()

    def load_analysis_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить файл для пакетного анализа",
            str(Path.cwd()),
            "Текстовые данные (*.csv *.txt *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx *.xls);;TXT (*.txt)",
        )
        if not path:
            return

        try:
            self.analysis_data_frame = self._read_dataset(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))
            return

        self.analysis_dataset_path = Path(path)
        self.results = []
        self._populate_analysis_columns()
        self._refresh_comparison_inputs()
        self.refresh_analysis()
        self.analyze_button.setEnabled(True)
        self.analysis_clear_button.setEnabled(True)
        self.analysis_dataset_label.setText(
            f"{self.analysis_dataset_path.name}: {format_int(len(self.analysis_data_frame))} строк, "
            f"{len(self.analysis_data_frame.columns)} колонок."
        )
        self.analysis_dataset_label.setToolTip(str(self.analysis_dataset_path))
        self._set_status("Данные анализа загружены", "ready")
        self._log(f"Файл анализа загружен: {self.analysis_dataset_path.name} ({len(self.analysis_data_frame)} строк).")
        self._switch_page(1)

    def clear_analysis_dataset(self) -> None:
        if self.analysis_data_frame.empty:
            return
        self.analysis_data_frame = pd.DataFrame()
        self.analysis_dataset_path = None
        self.results = []
        self.text_column_combo.clear()
        self.text_column_combo.addItem("text")
        self._refresh_comparison_inputs()
        self.analysis_dataset_label.setText("Файл для пакетного анализа не загружен.")
        self.analysis_dataset_label.setToolTip("")
        self.analyze_button.setEnabled(False)
        self.analysis_clear_button.setEnabled(False)
        self.populate_results_table([])
        self.refresh_analysis()
        self._set_status("Файл анализа очищен", "ready")
        self._log("Файл пакетного анализа очищен.")

    def load_val_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить validation-датасет",
            str(Path.cwd()),
            "Текстовые данные (*.csv *.txt *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx *.xls);;TXT (*.txt)",
        )
        if not path:
            return
        try:
            self.val_data_frame = self._read_dataset(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))
            return
        self.val_dataset_path = Path(path)
        self._refresh_comparison_inputs()
        self._refresh_dataset_summary()
        self.refresh_label_mapping_table()
        self._set_status("Данные загружены", "ready")
        self.statusBar().showMessage(
            f"Validation загружен: {self.val_dataset_path.name} ({format_int(len(self.val_data_frame))} строк)."
        )
        self._log(f"Validation загружен: {self.val_dataset_path.name} ({len(self.val_data_frame)} строк).")
        if self.preview_source != "train":
            self.populate_preview_table()

    def clear_val_dataset(self) -> None:
        if self.val_data_frame.empty:
            return
        self.val_data_frame = pd.DataFrame()
        self.val_dataset_path = None
        self._refresh_comparison_inputs()
        self._refresh_dataset_summary()
        self.refresh_label_mapping_table()
        if self.preview_source == "val":
            self.preview_offset = 0
            self.populate_preview_table()
        self._log("Validation-датасет очищен.")

    def load_test_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить test-датасет",
            str(Path.cwd()),
            "Текстовые данные (*.csv *.txt *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx *.xls);;TXT (*.txt)",
        )
        if not path:
            return
        try:
            self.test_data_frame = self._read_dataset(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))
            return
        self.test_dataset_path = Path(path)
        self._refresh_comparison_inputs()
        self._refresh_dataset_summary()
        self.refresh_label_mapping_table()
        self._set_status("Данные загружены", "ready")
        self.statusBar().showMessage(
            f"Test загружен: {self.test_dataset_path.name} ({format_int(len(self.test_data_frame))} строк)."
        )
        self._log(f"Test загружен: {self.test_dataset_path.name} ({len(self.test_data_frame)} строк).")
        if self.preview_source != "train":
            self.populate_preview_table()

    def clear_test_dataset(self) -> None:
        if self.test_data_frame.empty:
            return
        self.test_data_frame = pd.DataFrame()
        self.test_dataset_path = None
        self._refresh_comparison_inputs()
        self._refresh_dataset_summary()
        self.refresh_label_mapping_table()
        if self.preview_source == "test":
            self.preview_offset = 0
            self.populate_preview_table()
        self._log("Test-датасет очищен.")

    def _refresh_dataset_summary(self) -> None:
        train_loaded = not self.data_frame.empty
        val_loaded = not self.val_data_frame.empty
        test_loaded = not self.test_data_frame.empty

        if train_loaded and self.dataset_path is not None:
            self.train_status_label.setText(
                f"{self.dataset_path.name}: {format_int(len(self.data_frame))} строк, "
                f"{len(self.data_frame.columns)} колонок."
            )
            self.train_rows_card.set_values(format_int(len(self.data_frame)), "строк")
            self.dataset_columns_card.set_values(
                format_int(len(self.data_frame.columns)),
                "доступно для выбора",
            )
        else:
            self.train_status_label.setText(
                "Файл не открыт. Train обязателен для обучения."
            )
            self.train_rows_card.set_values("0", "строк")
            self.dataset_columns_card.set_values("0", "доступно для выбора")
        self.train_clear_button.setEnabled(train_loaded)

        if val_loaded and self.val_dataset_path is not None:
            self.val_status_label.setText(
                f"{self.val_dataset_path.name}: {format_int(len(self.val_data_frame))} строк, "
                f"{len(self.val_data_frame.columns)} колонок."
            )
            self.val_rows_card.set_values(format_int(len(self.val_data_frame)), "из отдельного файла")
        else:
            self.val_status_label.setText(
                "Не загружен. Validation будет отделена из train по проценту в настройках обучения."
            )
            self.val_rows_card.set_values("-", "не загружен")
        self.val_clear_button.setEnabled(val_loaded)

        if test_loaded and self.test_dataset_path is not None:
            self.test_status_label.setText(
                f"{self.test_dataset_path.name}: {format_int(len(self.test_data_frame))} строк, "
                f"{len(self.test_data_frame.columns)} колонок."
            )
            self.test_rows_card.set_values(format_int(len(self.test_data_frame)), "из отдельного файла")
        else:
            self.test_status_label.setText(
                "Не загружен. Test будет отделён из train по проценту в настройках обучения."
            )
            self.test_rows_card.set_values("-", "не загружен")
        self.test_clear_button.setEnabled(test_loaded)

        self._refresh_validation_split_state()
        self.populate_preview_table()
        self._update_context_bar()

    def _update_context_bar(self) -> None:
        if not hasattr(self, "context_dataset_label"):
            return

        if self.dataset_path is not None and not self.data_frame.empty:
            self.context_dataset_label.setText(self.dataset_path.name)
            self.context_dataset_label.setToolTip(str(self.dataset_path))
        else:
            self.context_dataset_label.setText("не загружен")
            self.context_dataset_label.setToolTip("")

        if not self.data_frame.empty:
            self.context_train_label.setText(format_int(len(self.data_frame)))
        else:
            self.context_train_label.setText("0")

        if not self.val_data_frame.empty:
            self.context_val_label.setText(format_int(len(self.val_data_frame)))
        else:
            self.context_val_label.setText("-")

        if not self.test_data_frame.empty:
            self.context_test_label.setText(format_int(len(self.test_data_frame)))
        else:
            self.context_test_label.setText("-")

        model_name = self.model_combo.currentText() if hasattr(self, "model_combo") else ""
        self.context_model_label.setText(self._format_context_model(model_name))
        self.context_model_label.setToolTip(model_name)

        device = self._analysis_device()
        self.context_device_label.setText(self._format_context_device(device))
        self.context_device_label.setToolTip(device)

    @staticmethod
    def _format_context_model(model_name: str) -> str:
        name = model_name.strip()
        if not name:
            return "-"
        path = Path(name)
        if path.exists() or name.startswith("models/") or "\\" in name or (":" in name and "/" not in name):
            return path.name
        return name

    @staticmethod
    def _format_context_device(device: str) -> str:
        value = (device or "").strip()
        if not value or value == "Auto":
            return "CPU"
        if value.startswith("CUDA"):
            return "GPU"
        return value

    def _current_preview_frame(self) -> pd.DataFrame:
        if self.preview_source == "val":
            return self.val_data_frame
        if self.preview_source == "test":
            return self.test_data_frame
        return self.data_frame

    def _on_preview_source_changed(self, _index: int) -> None:
        data = self.preview_source_combo.currentData()
        self.preview_source = str(data) if data is not None else "train"
        self.preview_offset = 0
        self.populate_preview_table()

    def _refresh_validation_split_state(self) -> None:
        if not hasattr(self, "train_val_split_spin"):
            return
        external_val = not self.val_data_frame.empty
        external_test = not self.test_data_frame.empty
        self.train_val_split_spin.setEnabled(not external_val)
        self.train_test_split_spin.setEnabled(not external_test)
        # split_seed / stratify влияют на оба сплита - оставляем активными, если хотя бы один сплит нужен
        any_split = not external_val or not external_test
        self.train_split_seed_spin.setEnabled(any_split)
        self.train_stratify_check.setEnabled(any_split)
        self.train_val_split_spin.setToolTip(
            "Валидационная выборка загружена из отдельного файла, процентное разбиение не применяется."
            if external_val
            else TRAINING_FIELD_HELP.get("Validation split", "")
        )
        self.train_test_split_spin.setToolTip(
            "Тестовая выборка загружена из отдельного файла, процентное разбиение не применяется."
            if external_test
            else TRAINING_FIELD_HELP.get("Test split", "")
        )

    def run_analysis(self) -> None:
        if self.analysis_data_frame.empty:
            QMessageBox.information(
                self,
                "Нет данных для анализа",
                "Загрузите отдельный файл для пакетного анализа на странице «Анализ».",
            )
            return
        text_column = self.text_column_combo.currentText()
        if text_column not in self.analysis_data_frame.columns:
            QMessageBox.warning(self, "Колонка не найдена", "Выберите колонку с текстом.")
            return

        source_column = self._guess_source_column(self.analysis_data_frame)
        options = self._analysis_options()
        self._set_status("Загрузка модели", "running")
        self.statusBar().showMessage("Загрузка модели...")
        QApplication.processEvents()

        try:
            analyzer = SentimentAnalyzer(self.model_combo.currentText(), options, self._analysis_device())
            self._log(f"Модель загружена: {self.model_combo.currentText()}.")
            self._set_status("Анализ выполняется", "running")
            self.statusBar().showMessage("Выполняется анализ...")
            QApplication.processEvents()
            self.results = analyzer.analyze(
                self.analysis_data_frame[text_column].fillna("").astype(str).tolist(),
                self.analysis_data_frame[source_column].fillna("").astype(str).tolist() if source_column else None,
            )
        except TransformerLoadError as exc:
            self._set_status("Ошибка", "error")
            self.statusBar().showMessage("Ошибка модели.")
            self._log(f"Ошибка модели: {exc}")
            QMessageBox.critical(self, "Ошибка модели", str(exc))
            return

        self._set_status("Анализ выполнен", "ready")
        self.statusBar().showMessage("Анализ завершен.")
        self._log(f"Анализ завершен. Обработано {len(self.results)} строк.")
        self.refresh_analysis()
        self._switch_page(1)

    def run_quick_text_analysis(self) -> None:
        text = self.quick_text_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Нет текста", "Введите текст для анализа.")
            return

        self._set_status("Анализ текста", "running")
        self.statusBar().showMessage("Выполняется быстрый анализ текста...")
        QApplication.processEvents()

        try:
            analyzer = SentimentAnalyzer(self.model_combo.currentText(), self._analysis_options(), self._analysis_device())
            result = analyzer.analyze([text])[0]
        except TransformerLoadError as exc:
            self._set_status("Ошибка", "error")
            self.statusBar().showMessage("Ошибка transformer-модели.")
            self._log(f"Ошибка быстрого анализа: {exc}")
            QMessageBox.critical(self, "Ошибка transformer-модели", str(exc))
            return

        self.quick_result_label.setText(f"{result.sentiment} · уверенность {result.confidence:.2f}")
        self._set_quick_result_sentiment(result.sentiment)
        self.quick_probability_label.setText("Распределение вероятностей")
        self.quick_confidence_bar.setValue(round(result.confidence * 100))
        self._populate_quick_probability_table(result.probabilities)
        self._set_status("Анализ выполнен", "ready")
        self.statusBar().showMessage("Быстрый анализ текста завершен.")
        self._log(f"Быстрый анализ: {result.sentiment}, уверенность {result.confidence:.2f}.")

    def clear_quick_text_analysis(self) -> None:
        self.quick_text_edit.clear()
        self.quick_result_label.setText("Результат появится здесь")
        self._set_quick_result_sentiment(None)
        self.quick_probability_label.setText("Вероятности появятся после анализа")
        self.quick_confidence_bar.setValue(0)
        self.quick_probability_table.setRowCount(0)

    def _populate_quick_probability_table(self, probabilities: dict[str, float]) -> None:
        rows = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        self.quick_probability_table.setRowCount(len(rows))
        for row, (label, score) in enumerate(rows):
            label_item = QTableWidgetItem(label)
            score_item = QTableWidgetItem(f"{score:.2f}")
            self.quick_probability_table.setItem(row, 0, label_item)
            self.quick_probability_table.setItem(row, 1, score_item)

    def _set_quick_result_sentiment(self, sentiment: str | None) -> None:
        state = {POSITIVE: "positive", NEUTRAL: "neutral", NEGATIVE: "negative"}.get(sentiment, "none")
        self.quick_result_label.setProperty("sentiment", state)
        self.quick_result_label.style().unpolish(self.quick_result_label)
        self.quick_result_label.style().polish(self.quick_result_label)

    def _analysis_options(self) -> PreprocessingOptions:
        return PreprocessingOptions(
            lowercase=self.lowercase_check.isChecked(),
            remove_punctuation=self.punctuation_check.isChecked(),
            remove_stop_words=self.stop_words_check.isChecked(),
            lemmatize=self.lemmatize_check.isChecked(),
        )

    def _analysis_device(self) -> str:
        if hasattr(self, "train_device_combo"):
            return self.train_device_combo.currentText()
        return "Auto"

    def export_report(self) -> None:
        if not self.results:
            QMessageBox.information(self, "Нет результатов", "Сначала выполните анализ.")
            return
        default_name = f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт отчёта", default_name, "HTML (*.html)")
        if not path:
            return
        report_path = export_html_report(path, self.results, self.model_combo.currentText())
        self.current_report_path = report_path
        self._load_report_preview(report_path)
        self._log(f"HTML-отчёт сохранен: {report_path}.")
        self.statusBar().showMessage(f"Отчёт сохранен: {report_path}")

    def preview_report(self, silent: bool = False) -> None:
        if not self.results:
            if not silent:
                QMessageBox.information(self, "Нет результатов", "Сначала выполните анализ.")
            return
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = export_html_report(
            reports_dir / "preview_report.html",
            self.results,
            self.model_combo.currentText(),
        )
        self.current_report_path = report_path
        self._load_report_preview(report_path)
        if not silent:
            self._log(f"Предпросмотр отчёта сформирован: {report_path}.")
            self.statusBar().showMessage("Предпросмотр отчёта сформирован.")

    def _load_report_preview(self, report_path: Path) -> None:
        if not hasattr(self, "report_view"):
            return
        html = report_path.read_text(encoding="utf-8")
        self.report_view.setSearchPaths([str(report_path.parent.resolve())])
        self.report_view.setHtml(html)
        if hasattr(self, "reports_stack"):
            self.reports_stack.setCurrentIndex(1)

    def open_report_in_browser(self) -> None:
        if self.current_report_path is None or not self.current_report_path.exists():
            QMessageBox.information(self, "Нет отчёта", "Сначала сформируйте предпросмотр или сохраните HTML-отчёт.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_report_path.resolve())))

    def browse_custom_model(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Папка с обученной моделью",
            str(Path.cwd()),
        )
        if not path:
            return

        folder = Path(path)
        if not (folder / "config.json").exists():
            QMessageBox.warning(
                self,
                "Не похоже на модель",
                "В выбранной папке нет config.json. Ожидаются файлы формата Hugging Face "
                "(config.json, model.safetensors или pytorch_model.bin, tokenizer_config.json, "
                "special_tokens_map.json и т. п.).",
            )
            return

        from app.models.sentiment import WEIGHT_FILES, TOKENIZER_VOCAB_FILES

        weights_present = any((folder / name).exists() for name in WEIGHT_FILES)
        if not weights_present:
            QMessageBox.warning(
                self,
                "Не найдены веса",
                "В папке есть config.json, но нет файлов весов "
                "(model.safetensors или pytorch_model.bin). "
                "Модель не загрузится без них.",
            )
            return

        vocab_present = any((folder / name).exists() for name in TOKENIZER_VOCAB_FILES)
        if not vocab_present:
            QMessageBox.warning(
                self,
                "Нет словаря токенайзера",
                "В папке нет файла словаря токенайзера: tokenizer.json (fast), "
                "vocab.txt (BERT/DistilBERT), spiece.model / sentencepiece.bpe.model (XLM-R/T5) "
                "или tokenizer.model. Без него токенайзер не сможет загрузиться - "
                "скопируйте недостающий файл из папки базовой модели.",
            )
            return

        path_str = str(folder)
        if self.model_combo.findText(path_str) == -1:
            self.model_combo.addItem(path_str)
        self.model_combo.setCurrentText(path_str)
        self._refresh_model_lists(path_str)
        self._log(f"Подключена локальная модель: {path_str}.")
        self.statusBar().showMessage(f"Локальная модель выбрана: {folder.name}")

    def select_training_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Папка для сохранения модели", str(Path.cwd() / "models"))
        if path:
            self.train_output_edit.setText(path)

    def start_training(self) -> None:
        if self.data_frame.empty:
            QMessageBox.information(self, "Нет данных", "Сначала загрузите датасет с текстом и метками.")
            return

        text_column = self.train_text_column_combo.currentText()
        label_column = self.train_label_column_combo.currentText()
        if text_column not in self.data_frame.columns or label_column not in self.data_frame.columns:
            QMessageBox.warning(self, "Колонки не выбраны", "Выберите колонку текста и колонку метки.")
            return
        if text_column == label_column:
            QMessageBox.warning(self, "Некорректные колонки", "Колонка текста и колонка метки должны отличаться.")
            return

        prepared = self._prepare_training_data()
        if prepared.used_rows == 0:
            QMessageBox.warning(
                self,
                "Нет строк для обучения",
                "После фильтрации меток не осталось строк. Проверьте таблицу соответствия меток.",
            )
            return

        if len(prepared.class_counts) < 2:
            QMessageBox.warning(self, "Недостаточно классов", "Для обучения нужно минимум два класса после настройки меток.")
            return

        val_prepared: LabelPrepareResult | None = None
        if not self.val_data_frame.empty:
            if text_column not in self.val_data_frame.columns or label_column not in self.val_data_frame.columns:
                QMessageBox.warning(
                    self,
                    "Валидационная выборка несовместима",
                    "В отдельном validation-файле нет тех же колонок текста и метки, что и в train.",
                )
                return
            val_prepared = self._prepare_training_data(self.val_data_frame)
            if val_prepared.used_rows == 0:
                QMessageBox.warning(
                    self,
                    "Валидационная выборка пуста после фильтрации",
                    "После применения настройки меток в validation-файле не осталось строк.",
                )
                return

        test_prepared: LabelPrepareResult | None = None
        if not self.test_data_frame.empty:
            if text_column not in self.test_data_frame.columns or label_column not in self.test_data_frame.columns:
                QMessageBox.warning(
                    self,
                    "Тестовая выборка несовместима",
                    "В отдельном test-файле нет тех же колонок текста и метки, что и в train.",
                )
                return
            test_prepared = self._prepare_training_data(self.test_data_frame)
            if test_prepared.used_rows == 0:
                QMessageBox.warning(
                    self,
                    "Тестовая выборка пуста после фильтрации",
                    "После применения настройки меток в test-файле не осталось строк.",
                )
                return

        config = TrainConfig(
            model_name=self.train_base_model_combo.currentText(),
            output_dir=Path(self.train_output_edit.text()).resolve(),
            experiment_name=self.train_experiment_name_edit.text().strip() or "sentiment_experiment",
            run_description=self.train_description_edit.toPlainText().strip(),
            epochs=self.train_epochs_spin.value(),
            batch_size=self.train_batch_spin.value(),
            eval_batch_size=self.train_eval_batch_spin.value(),
            learning_rate=self.train_lr_spin.value(),
            weight_decay=self.train_weight_decay_spin.value(),
            validation_split=self.train_val_split_spin.value(),
            test_split=self.train_test_split_spin.value(),
            stratify_split=self.train_stratify_check.isChecked(),
            split_seed=self.train_split_seed_spin.value(),
            seed=self.train_seed_spin.value(),
            max_length=self.train_max_length_spin.value(),
            padding=self.train_padding_combo.currentText(),
            truncation=self.train_truncation_check.isChecked(),
            use_fast_tokenizer=self.train_fast_tokenizer_check.isChecked(),
            pad_to_multiple_of=self.train_pad_multiple_spin.value() or None,
            device=self.train_device_combo.currentText(),
            local_files_only=self.train_local_only_check.isChecked(),
            trust_remote_code=self.train_trust_remote_check.isChecked(),
            ignore_mismatched_sizes=self.train_ignore_mismatch_check.isChecked(),
            problem_type=self.train_problem_type_combo.currentText(),
            gradient_accumulation_steps=self.train_grad_accum_spin.value(),
            max_grad_norm=self.train_max_grad_norm_spin.value(),
            optimizer=self.train_optimizer_combo.currentText(),
            adam_beta1=self.train_beta1_spin.value(),
            adam_beta2=self.train_beta2_spin.value(),
            adam_epsilon=self.train_epsilon_spin.value(),
            lr_scheduler_type=self.train_scheduler_combo.currentText(),
            warmup_ratio=self.train_warmup_ratio_spin.value(),
            warmup_steps=self.train_warmup_steps_spin.value(),
            label_smoothing_factor=self.train_label_smoothing_spin.value(),
            class_weights=self.train_class_weights_combo.currentText(),
            freeze_base_model=self.train_freeze_base_check.isChecked(),
            freeze_embeddings=self.train_freeze_embeddings_check.isChecked(),
            freeze_encoder_layers=self.train_freeze_layers_spin.value(),
            train_classifier_only=self.train_freeze_base_check.isChecked(),
            fp16=self.train_fp16_check.isChecked(),
            gradient_checkpointing=self.train_gradient_checkpointing_check.isChecked(),
            dataloader_num_workers=self.train_workers_spin.value(),
            dataloader_pin_memory=self.train_pin_memory_check.isChecked(),
            drop_duplicates=self.train_drop_duplicates_check.isChecked(),
            max_samples=self.train_max_samples_spin.value(),
            shuffle_dataset=self.train_shuffle_check.isChecked(),
            early_stopping=self.train_early_stop_check.isChecked(),
            early_stopping_patience=self.train_early_patience_spin.value(),
            metric_for_best_model=self.train_metric_combo.currentText(),
            save_predictions=self.train_save_predictions_check.isChecked(),
            save_training_config=self.train_save_config_check.isChecked(),
        )

        self.training_log.clear()
        self.training_log_stack.setCurrentIndex(1)
        self.training_metrics_chart.empty("Обучение выполняется")
        self._append_training_log("Данные", "section")
        self._append_training_log(
            f"Train после меток: {format_int(prepared.used_rows)} строк; "
            f"исключено {format_int(prepared.excluded_rows)}."
        )
        self._append_training_log(
            "Классы: "
            + " · ".join(f"{label}: {format_int(count)}" for label, count in prepared.class_counts.items())
        )
        self._append_split_plan_log(val_prepared, test_prepared, config)
        if config.max_samples > 0:
            self._append_training_log(
                f"Лимит train: {format_int(config.max_samples)} строк после очистки/перемешивания."
            )
        if val_prepared is not None:
            self._append_training_log(
                f"Validation файл: {format_int(val_prepared.used_rows)} строк; "
                f"исключено {format_int(val_prepared.excluded_rows)}."
            )
        if test_prepared is not None:
            self._append_training_log(
                f"Test файл: {format_int(test_prepared.used_rows)} строк; "
                f"исключено {format_int(test_prepared.excluded_rows)}."
            )
        self.training_progress.setRange(0, 0)
        self.train_button.setEnabled(False)
        self.cancel_train_button.setEnabled(True)
        self._set_status("Обучение выполняется", "running")
        self.statusBar().showMessage("Выполняется дообучение модели...")

        self.training_thread = TrainingThread(
            prepared.texts,
            prepared.labels,
            config,
            val_texts=val_prepared.texts if val_prepared is not None else None,
            val_labels=val_prepared.labels if val_prepared is not None else None,
            test_texts=test_prepared.texts if test_prepared is not None else None,
            test_labels=test_prepared.labels if test_prepared is not None else None,
        )
        self.training_thread.message.connect(self.append_training_message)
        self.training_thread.completed.connect(self.on_training_completed)
        self.training_thread.failed.connect(self.on_training_failed)
        self.training_thread.canceled.connect(self.on_training_canceled)
        self.training_thread.start()

    def on_training_completed(self, result: TrainResult) -> None:
        self.training_progress.setRange(0, 1)
        self.training_progress.setValue(1)
        self.train_button.setEnabled(True)
        self.cancel_train_button.setEnabled(False)
        self._set_status("Обучение завершено", "ready")
        val_source = "файл" if result.validation_source == "external" else "split"
        self.training_metrics_chart.draw_training_result(result)
        self._append_training_log("Итог", "section")
        self._append_training_log(
            f"Validation ({val_source}): accuracy {result.accuracy:.3f} · macro F1 {result.macro_f1:.3f}"
        )
        if result.test_size > 0:
            test_source = "файл" if result.test_source == "external" else "split"
            self._append_training_log(
                f"Test ({test_source}, {format_int(result.test_size)} строк): "
                f"accuracy {result.test_accuracy:.3f} · macro F1 {result.test_macro_f1:.3f}"
            )
        output_path = str(result.output_dir)
        if self.model_combo.findText(output_path) == -1:
            self.model_combo.addItem(output_path)
        self.model_combo.setCurrentText(output_path)
        self._refresh_model_lists(output_path)
        self.statusBar().showMessage(f"Модель обучена и сохранена: {output_path}")
        self._log(f"Обучение завершено. Модель сохранена: {output_path}.")

    def on_training_failed(self, message: str) -> None:
        self.training_progress.setRange(0, 1)
        self.training_progress.setValue(0)
        self.train_button.setEnabled(True)
        self.cancel_train_button.setEnabled(False)
        self._set_status("Ошибка", "error")
        self._append_training_log("Ошибка обучения", "section")
        self._append_training_log(message)
        self.statusBar().showMessage("Ошибка обучения.")
        QMessageBox.critical(self, "Ошибка обучения", message)

    def cancel_training(self) -> None:
        if self.training_thread is None or not self.training_thread.isRunning():
            return
        self.training_thread.request_stop()
        self.cancel_train_button.setEnabled(False)
        self._set_status("Остановка", "running")
        self._append_training_log("Остановка", "section")
        self._append_training_log("Запрошено прерывание. Текущий batch будет завершён, затем обучение остановится.")
        self.statusBar().showMessage("Остановка обучения...")

    def on_training_canceled(self, message: str) -> None:
        self.training_progress.setRange(0, 1)
        self.training_progress.setValue(0)
        self.train_button.setEnabled(True)
        self.cancel_train_button.setEnabled(False)
        self._set_status("Прервано", "ready")
        self._append_training_log("Обучение прервано", "section")
        self._append_training_log(message or "Остановлено пользователем.")
        self.statusBar().showMessage("Обучение прервано пользователем.")

    def append_training_message(self, message: str) -> None:
        text = message.strip()
        if not text:
            return
        if text.startswith("Фактическая выборка"):
            self._append_training_log(text.replace("Фактическая выборка: ", "Размеры: "), "metric")
            return
        elif text.startswith("Источник модели"):
            self._append_training_log(text.replace("Источник модели: ", "Источник: "))
            return
        if text.startswith("Загрузка базовой модели"):
            self._append_training_log("Модель", "section")
            self._append_training_log(text.replace("Загрузка базовой модели: ", "База: "))
        elif text.startswith("Токенизация"):
            self._append_training_log("Эпохи", "section")
            self.training_log_stack.setCurrentIndex(1)
            self.training_log.appendPlainText("  epoch   loss     acc    macro F1")
        elif text.startswith("Эпоха"):
            self._append_training_log(self._format_epoch_log(text), "metric")
        elif text.startswith("Held-out"):
            self._append_training_log("Test", "section")
        elif text.startswith("Test:"):
            self._append_training_log(self._format_test_log(text), "metric")
        elif text.startswith("Модель сохранена"):
            self._append_training_log("Сохранено", "section")
            self._append_training_log(text.replace("Модель сохранена: ", "Папка: "))
        else:
            self._append_training_log(text)

    def _append_training_log(self, text: str, kind: str = "info") -> None:
        if hasattr(self, "training_log_stack"):
            self.training_log_stack.setCurrentIndex(1)
        if kind == "section":
            if self.training_log.toPlainText():
                self.training_log.appendPlainText("")
            self.training_log.appendPlainText(text)
            return
        prefix = "  " if kind == "metric" else "  "
        self.training_log.appendPlainText(f"{prefix}{text}")

    @staticmethod
    def _format_epoch_log(text: str) -> str:
        match = re.search(
            r"Эпоха\s+(\d+/\d+):\s+loss=([0-9.]+),\s+accuracy=([0-9.]+),\s+macro_f1=([0-9.]+)",
            text,
        )
        if not match:
            return text
        epoch, loss, accuracy, macro_f1 = match.groups()
        return f"{epoch:<7} {loss:<8} {accuracy:<6} {macro_f1}"

    @staticmethod
    def _format_test_log(text: str) -> str:
        match = re.search(r"accuracy=([0-9.]+),\s+macro_f1=([0-9.]+)", text)
        if not match:
            return text
        accuracy, macro_f1 = match.groups()
        return f"accuracy {accuracy} · macro F1 {macro_f1}"

    def _append_split_plan_log(
        self,
        val_prepared: LabelPrepareResult | None,
        test_prepared: LabelPrepareResult | None,
        config: TrainConfig,
    ) -> None:
        has_val_file = val_prepared is not None
        has_test_file = test_prepared is not None
        if has_val_file and has_test_file:
            self._append_training_log("Разбиение: train не делится; validation и test из файлов.")
        elif has_val_file:
            if config.test_split > 0:
                self._append_training_log(
                    f"Разбиение: validation из файла; test {config.test_split:.0%} из train."
                )
            else:
                self._append_training_log("Разбиение: validation из файла; test не используется.")
        elif has_test_file:
            self._append_training_log(
                f"Разбиение: test из файла; validation {config.validation_split:.0%} из train."
            )
        elif config.test_split > 0:
            self._append_training_log(
                f"Разбиение: test {config.test_split:.0%} из train; "
                f"validation {config.validation_split:.0%} из оставшейся части."
            )
        else:
            self._append_training_log(
                f"Разбиение: validation {config.validation_split:.0%} из train; test не используется."
            )

    def _label_action_values(self) -> list[str]:
        if self.target_scheme_combo.currentText() == SENTIMENT_SCHEME:
            return [EXCLUDE_LABEL, POSITIVE, NEUTRAL, NEGATIVE]
        return [KEEP_ORIGINAL, EXCLUDE_LABEL]

    def _default_label_action(self, raw_label: str) -> str:
        action = auto_target_for_label(raw_label)
        if self.target_scheme_combo.currentText() == SENTIMENT_SCHEME:
            return action if action in TARGET_CLASSES or action == EXCLUDE_LABEL else EXCLUDE_LABEL
        return EXCLUDE_LABEL if action == EXCLUDE_LABEL else KEEP_ORIGINAL

    def refresh_label_mapping_table(self) -> None:
        if not hasattr(self, "label_mapping_table"):
            return

        current_mapping = self._current_label_mapping()
        self.label_action_combos.clear()
        self.label_target_edits.clear()
        self.label_mapping_table.setRowCount(0)
        self.label_count_by_token = Counter()

        if self.data_frame.empty or not hasattr(self, "train_label_column_combo"):
            self.label_mapping_summary.setText("Загрузите датасет и выберите колонку метки.")
            return

        label_column = self.train_label_column_combo.currentText()
        if label_column not in self.data_frame.columns:
            self.label_mapping_summary.setText("Колонка метки не выбрана или отсутствует в датасете.")
            return

        parse_as_list = self.label_format_combo.currentText() == "Список меток в строке"
        for frame in (self.data_frame, self.val_data_frame, self.test_data_frame):
            if frame.empty or label_column not in frame.columns:
                continue
            for value in frame[label_column].tolist():
                for token in parse_label_tokens(value, parse_as_list=parse_as_list):
                    self.label_count_by_token[token] += 1

        rows = sorted(self.label_count_by_token.items(), key=lambda item: (-item[1], label_key(item[0])))
        self.label_mapping_table.setHorizontalHeaderLabels(["Метка", "Кол-во", "Действие", "Новая метка"])
        self.label_mapping_table.setRowCount(len(rows))
        for row, (raw_label, count) in enumerate(rows):
            self.label_mapping_table.setItem(row, 0, QTableWidgetItem(raw_label))
            self.label_mapping_table.setItem(row, 1, QTableWidgetItem(format_int(count)))

            config = current_mapping.get(raw_label, {"action": self._default_label_action(raw_label), "custom": ""})
            action = str(config.get("action") or self._default_label_action(raw_label))
            combo = make_combo(self._label_action_values(), action)
            combo.setToolTip("Выберите действие для исходной метки.")
            combo.currentTextChanged.connect(self.update_label_mapping_summary)

            custom_edit = QLineEdit(str(config.get("custom") or ""))
            custom_edit.setPlaceholderText("Например: positive")
            custom_edit.editingFinished.connect(self.update_label_mapping_summary)

            self.label_action_combos[raw_label] = combo
            self.label_target_edits[raw_label] = custom_edit
            self.label_mapping_table.setCellWidget(row, 2, combo)
            self.label_mapping_table.setCellWidget(row, 3, custom_edit)
            self.label_mapping_table.setRowHeight(row, 44)

        self.update_label_mapping_summary()

    def auto_configure_label_mapping(self) -> None:
        if not hasattr(self, "label_mapping_table"):
            return
        for raw_label, combo in self.label_action_combos.items():
            action = self._default_label_action(raw_label)
            combo.setCurrentText(action)
            target_edit = self.label_target_edits.get(raw_label)
            if target_edit is not None:
                target_edit.clear()
        self.update_label_mapping_summary()

    def _current_label_mapping(self) -> dict[str, dict[str, str]]:
        mapping: dict[str, dict[str, str]] = {}
        for raw_label, combo in self.label_action_combos.items():
            target_edit = self.label_target_edits.get(raw_label)
            mapping[raw_label] = {
                "action": combo.currentText().strip() or self._default_label_action(raw_label),
                "custom": target_edit.text().strip() if target_edit is not None else "",
            }
        return mapping

    def _map_label_value(self, raw_value: object) -> str | None:
        tokens = parse_label_tokens(
            raw_value,
            parse_as_list=self.label_format_combo.currentText() == "Список меток в строке",
        )
        return self._map_tokens_to_target(tokens, self._current_label_mapping())

    def _map_tokens_to_target(self, tokens: list[str], mapping: dict[str, dict[str, str]]) -> str | None:
        if not tokens:
            return None

        strategy = self.multilabel_strategy_combo.currentText()
        parse_as_list = self.label_format_combo.currentText() == "Список меток в строке"

        if parse_as_list and len(tokens) > 1 and strategy == "Исключать строки со списком меток":
            return None

        if parse_as_list and len(tokens) > 1 and strategy == "Первая метка в списке":
            tokens = tokens[:1]

        for token in tokens:
            config = mapping.get(token)
            action = str(config.get("action") if config else auto_target_for_label(token))
            custom_target = str(config.get("custom") or "").strip() if config else ""
            if custom_target:
                return custom_target
            if action == EXCLUDE_LABEL:
                continue
            if action == KEEP_ORIGINAL:
                return token
            return action
        return None

    def _prepare_training_data(
        self,
        frame: pd.DataFrame | None = None,
    ) -> LabelPrepareResult:
        source = self.data_frame if frame is None else frame
        text_column = self.train_text_column_combo.currentText()
        label_column = self.train_label_column_combo.currentText()
        parse_as_list = self.label_format_combo.currentText() == "Список меток в строке"
        mapping = self._current_label_mapping()

        texts: list[str] = []
        labels: list[str] = []
        excluded = 0

        if source is None or source.empty or text_column not in source.columns or label_column not in source.columns:
            return LabelPrepareResult(texts=[], labels=[], used_rows=0, excluded_rows=0, class_counts=Counter())

        for _, row in source.iterrows():
            text = str(row.get(text_column, "")).strip()
            tokens = parse_label_tokens(row.get(label_column, ""), parse_as_list=parse_as_list)
            target = self._map_tokens_to_target(tokens, mapping)
            if not text or target is None:
                excluded += 1
                continue
            texts.append(text)
            labels.append(str(target))

        return LabelPrepareResult(texts=texts, labels=labels, used_rows=len(texts), excluded_rows=excluded, class_counts=Counter(labels))

    def update_label_mapping_summary(self) -> None:
        if not hasattr(self, "label_mapping_summary"):
            return
        if self.data_frame.empty or not self.label_action_combos:
            self.label_mapping_summary.setText("Загрузите датасет и выберите колонку метки.")
            return

        prepared = self._prepare_training_data()
        counts = "; ".join(f"{label}: {format_int(count)}" for label, count in prepared.class_counts.items()) or "нет классов"
        list_note = ""
        if self.label_format_combo.currentText() == "Список меток в строке":
            list_note = " Режим обучения сейчас single-label: из списка меток для каждой строки выбирается одна целевая метка."

        extras: list[str] = []
        if not self.val_data_frame.empty:
            val_prepared = self._prepare_training_data(self.val_data_frame)
            extras.append(
                f"Validation (файл): {format_int(val_prepared.used_rows)} строк, исключено {format_int(val_prepared.excluded_rows)}"
            )
        if not self.test_data_frame.empty:
            test_prepared = self._prepare_training_data(self.test_data_frame)
            extras.append(
                f"Test (файл): {format_int(test_prepared.used_rows)} строк, исключено {format_int(test_prepared.excluded_rows)}"
            )
        extras_text = (" " + " · ".join(extras) + ".") if extras else ""

        self.label_mapping_summary.setText(
            f"Train после настройки меток: использовано {format_int(prepared.used_rows)} строк, "
            f"исключено {format_int(prepared.excluded_rows)} строк. Классы: {counts}.{list_note}{extras_text}"
        )

    def refresh_analysis(self) -> None:
        summary = summarize_results(self.results)
        total = len(self.analysis_data_frame)
        processed = int(summary["processed"])
        uncertain_examples = self._active_learning_candidates()
        self.total_metric_label.setText(f"Всего: {format_int(total)}")
        self.processed_metric_label.setText(
            f"Проанализировано: {format_int(processed)} ({processed / max(total, 1):.1%})"
        )
        self.confidence_metric_label.setText(f"Средняя уверенность: {float(summary['avg_confidence']):.2f}")
        self.low_confidence_metric_label.setText(
            f"На разметку: {format_int(len(uncertain_examples))} (<{ACTIVE_LEARNING_THRESHOLD:.2f})"
        )
        self.populate_results_table(self.results)
        self.class_chart.draw_class_distribution(class_distribution(self.results))
        self.confidence_chart.draw_confidence_histogram(self.results)
        self.refresh_monitoring()
        self._refresh_workflow_state()
        if self.results and hasattr(self, "report_view"):
            self.preview_report(silent=True)

    def refresh_monitoring(self) -> None:
        report = build_drift_report(self.results)
        self.drift_chart.draw_drift(self.results)
        self.monitoring_summary_label.setText(report.message)
        counts = class_distribution(self.results)
        total = len(self.results)
        avg_confidence = sum(result.confidence for result in self.results) / total if total else 0
        rows = [
            ("Проанализировано", format_int(total)),
            ("Средняя уверенность", f"{avg_confidence:.2f}"),
            (
                f"Кандидаты для разметки (<{ACTIVE_LEARNING_THRESHOLD:.2f})",
                format_int(len(self._active_learning_candidates())),
            ),
        ]
        rows.extend((f"Класс: {label}", format_int(count)) for label, count in counts.most_common(8))
        self.monitoring_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.monitoring_table.setItem(row, 0, QTableWidgetItem(name))
            self.monitoring_table.setItem(row, 1, QTableWidgetItem(value))

    def _active_learning_candidates(self) -> list[UncertainExample]:
        return select_uncertain_examples(
            [result.text for result in self.results],
            [result.probabilities for result in self.results],
            threshold=ACTIVE_LEARNING_THRESHOLD,
        )

    def populate_preview_table(self) -> None:
        source_frame = self._current_preview_frame()
        total = len(source_frame)
        page_size = self.preview_page_size_spin.value() if hasattr(self, "preview_page_size_spin") else 200
        if total == 0:
            self.preview_offset = 0
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            if hasattr(self, "preview_range_label"):
                self.preview_range_label.setText("Строки 0-0 из 0")
                self.preview_jump_spin.setRange(1, 1)
                for button in (self.preview_first_button, self.preview_prev_button, self.preview_next_button):
                    button.setEnabled(False)
            return

        self.preview_offset = max(0, min(self.preview_offset, max(total - 1, 0)))
        frame = source_frame.iloc[self.preview_offset : self.preview_offset + page_size]
        self.preview_table.setRowCount(len(frame))
        self.preview_table.setColumnCount(len(frame.columns))
        self.preview_table.setHorizontalHeaderLabels([str(column) for column in frame.columns])
        header = self.preview_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for column_index in range(len(frame.columns)):
            if self.preview_table.columnWidth(column_index) < 120:
                self.preview_table.setColumnWidth(column_index, 120)
        if len(frame.columns) > 0:
            preferred = self.text_column_combo.currentText()
            preferred_index = list(frame.columns).index(preferred) if preferred in frame.columns else 0
            self.preview_table.setColumnWidth(preferred_index, max(self.preview_table.columnWidth(preferred_index), 320))
        for row_index, (_, row) in enumerate(frame.iterrows()):
            for column_index, value in enumerate(row):
                self.preview_table.setItem(row_index, column_index, QTableWidgetItem(str(value)[:300]))
        self._rebalance_table_columns(self.preview_table)
        start = self.preview_offset + 1
        end = self.preview_offset + len(frame)
        if hasattr(self, "preview_range_label"):
            self.preview_range_label.setText(f"Строки {format_int(start)}-{format_int(end)} из {format_int(total)}")
            self.preview_jump_spin.blockSignals(True)
            self.preview_jump_spin.setRange(1, total)
            self.preview_jump_spin.setValue(start)
            self.preview_jump_spin.blockSignals(False)
            self.preview_first_button.setEnabled(self.preview_offset > 0)
            self.preview_prev_button.setEnabled(self.preview_offset > 0)
            self.preview_next_button.setEnabled(end < total)

    def _reset_preview_page(self) -> None:
        self.preview_offset = 0
        self.populate_preview_table()

    def _jump_preview_page(self) -> None:
        if self._current_preview_frame().empty:
            return
        self.preview_offset = self.preview_jump_spin.value() - 1
        self.populate_preview_table()

    def _preview_first_page(self) -> None:
        self.preview_offset = 0
        self.populate_preview_table()

    def _preview_prev_page(self) -> None:
        page_size = self.preview_page_size_spin.value()
        self.preview_offset = max(0, self.preview_offset - page_size)
        self.populate_preview_table()

    def _preview_next_page(self) -> None:
        source = self._current_preview_frame()
        if source.empty:
            return
        page_size = self.preview_page_size_spin.value()
        self.preview_offset = min(max(len(source) - 1, 0), self.preview_offset + page_size)
        self.populate_preview_table()

    def populate_results_table(self, results: list[AnalysisResult]) -> None:
        visible_results = results[:1000]
        self.result_count_label.setText(f"Всего: {format_int(len(results))}")
        score_labels = result_score_labels(visible_results)
        headers = ["#", "Текст", "Класс", "Увер.", *[compact_score_label(label) for label in score_labels]]
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        for offset, label in enumerate(score_labels, start=4):
            header_item = self.results_table.horizontalHeaderItem(offset)
            if header_item is not None:
                header_item.setToolTip(label)
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.results_table.setColumnWidth(0, 45)
        self.results_table.setColumnWidth(1, 520)
        self.results_table.setColumnWidth(2, 128)
        self.results_table.setColumnWidth(3, 80)
        for column in range(4, 4 + len(score_labels)):
            self.results_table.setColumnWidth(column, 76)
        self.results_table.setRowCount(len(visible_results))
        for row, result in enumerate(visible_results):
            values = [
                str(row + 1),
                result.text[:260],
                result.sentiment,
                f"{result.confidence:.2f}",
                *[f"{result.probabilities.get(label, 0.0):.2f}" for label in score_labels],
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 2:
                    item.setForeground(QColor(SENTIMENT_COLORS.get(result.sentiment, "#111827")))
                self.results_table.setItem(row, column, item)
        self._rebalance_table_columns(self.results_table)

    def populate_profile_table(self) -> None:
        if not hasattr(self, "profile_table"):
            return
        rows = comparison_rows()
        known = {self._model_identity(str(profile["name"])) for profile in rows}
        for model_path in self._existing_local_models():
            identity = self._model_identity(model_path)
            if identity not in known:
                rows.append(self._local_model_profile(model_path))
                known.add(identity)
        if hasattr(self, "model_combo"):
            for index in range(self.model_combo.count()):
                model_name = self.model_combo.itemText(index)
                identity = self._model_identity(model_name)
                if identity not in known and Path(model_name).exists():
                    rows.append(self._local_model_profile(model_name))
                    known.add(identity)

        deduped_rows: list[dict[str, float | int | str | bool]] = []
        seen_registry_keys: set[str] = set()
        for profile in rows:
            registry_key = str(profile.get("registry_key") or self._model_identity(str(profile["name"])))
            if registry_key in seen_registry_keys:
                continue
            seen_registry_keys.add(registry_key)
            deduped_rows.append(profile)

        rows = sorted(
            deduped_rows,
            key=lambda profile: (
                0 if str(profile.get("status")) == "локальная обученная" else 1,
                str(profile.get("display_name") or profile.get("name") or "").casefold(),
            ),
        )

        self._populating_profile_table = True
        try:
            self.profile_table.clearContents()
            self.profile_table.setRowCount(len(rows))
            for row, profile in enumerate(rows):
                name = str(profile["name"])
                display_name = str(profile.get("display_name") or self._display_model_name(name))
                status = str(profile.get("status") or self._model_status(name))
                can_rename = bool(profile.get("can_rename"))
                values = [
                    display_name,
                    status,
                    str(profile.get("label_schema") or "-"),
                    str(profile.get("quality_group") or "-"),
                    self._format_profile_metric(profile.get("accuracy")),
                    self._format_profile_metric(profile.get("macro_f1")),
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column == 0:
                        item.setData(Qt.ItemDataRole.UserRole, name)
                        item.setToolTip(name)
                        if can_rename:
                            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                    if hasattr(self, "model_combo") and name == self.model_combo.currentText():
                        item.setBackground(QColor("#d7e5fb"))
                    self.profile_table.setItem(row, column, item)
                self.profile_table.setRowHeight(row, self._model_action_row_height())
                self.profile_table.setCellWidget(row, 6, self._build_model_action_cell(name, can_rename, status))
        finally:
            self._populating_profile_table = False

    def _selected_profile_model_path(self) -> Path | None:
        if not hasattr(self, "profile_table"):
            return None
        current_row = self.profile_table.currentRow()
        if current_row < 0:
            return None
        item = self.profile_table.item(current_row, 0)
        if item is None:
            return None
        path = Path(str(item.data(Qt.ItemDataRole.UserRole) or item.text()))
        if not path.exists() or not path.is_dir():
            return None
        return path

    def _refresh_model_lists(self, selected_model: str | None = None) -> None:
        self._refresh_inference_model_choices(selected_model)
        self.populate_profile_table()
        self.populate_comparison_model_table()
        self._update_context_bar()

    def _refresh_inference_model_choices(self, selected_model: str | None = None) -> None:
        if not hasattr(self, "model_combo"):
            return
        current_text = selected_model or self.model_combo.currentText()
        models = self._inference_local_models()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for model_path in models:
            self.model_combo.addItem(model_path)
        if current_text and self.model_combo.findText(current_text) == -1 and Path(current_text).exists():
            self.model_combo.addItem(current_text)
        if current_text:
            self.model_combo.setCurrentText(current_text)
        self.model_combo.blockSignals(False)

    def rename_selected_model(self, model_name: str | None = None) -> None:
        path = Path(model_name) if model_name else self._selected_profile_model_path()
        if path is None:
            QMessageBox.information(self, "Нет модели", "Выберите локальную модель в таблице.")
            return
        if not self._is_trained_inference_model(path):
            QMessageBox.information(
                self,
                "Переименование недоступно",
                "Базовые модели для обучения нельзя переименовывать из интерфейса.",
            )
            return

        row = self._find_profile_row(str(path.resolve()))
        if row is None:
            return
        item = self.profile_table.item(row, 0)
        if item is None:
            return

        self._pending_profile_rename_path = str(path.resolve())
        self._pending_profile_rename_original_name = item.text().strip()
        self.profile_table.setCurrentCell(row, 0)
        self.profile_table.editItem(item)

    def delete_selected_model(self, model_name: str | None = None) -> None:
        path = Path(model_name) if model_name else self._selected_profile_model_path()
        if path is None:
            QMessageBox.information(self, "Нет модели", "Выберите локальную модель в таблице.")
            return

        reply = QMessageBox.question(
            self,
            "Удалить модель",
            f"Удалить папку модели '{path.name}' со всеми файлами?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        current_model = self.model_combo.currentText() if hasattr(self, "model_combo") else ""
        next_model = "" if Path(current_model) == path else current_model
        try:
            shutil.rmtree(path)
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось удалить", str(exc))
            return

        self._refresh_model_lists(next_model)
        self.statusBar().showMessage(f"Модель удалена: {path.name}")
        self._log(f"Модель удалена: {path}.")

    def _build_model_action_cell(self, model_name: str, can_rename: bool, status: str) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        path = Path(model_name)
        button_height = self._model_action_button_height()
        delete_button = QPushButton("Удалить")
        delete_button.setObjectName("tableDangerActionButton")
        delete_button.setFixedHeight(button_height)
        delete_button.setEnabled(path.exists() and path.is_dir())
        delete_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        delete_button.clicked.connect(lambda checked=False, name=model_name: self.delete_selected_model(name))

        if can_rename:
            rename_button = QPushButton("Переим.")
            rename_button.setObjectName("tableActionButton")
            rename_button.setFixedHeight(button_height)
            rename_button.setEnabled(path.exists() and path.is_dir())
            rename_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            rename_button.clicked.connect(lambda checked=False, name=model_name: self.rename_selected_model(name))
            layout.addWidget(rename_button)
        layout.addWidget(delete_button)
        widget.setMinimumHeight(self._model_action_row_height() - 4)
        return widget

    def _model_action_button_height(self) -> int:
        return max(36, self.fontMetrics().height() + 18)

    def _model_action_row_height(self) -> int:
        return self._model_action_button_height() + 18

    @staticmethod
    def _display_model_name(model_name: str) -> str:
        path = Path(model_name)
        if path.name and (path.exists() or any(sep in model_name for sep in ("\\", "/"))):
            return path.name
        return model_name

    def populate_comparison_model_table(self) -> None:
        if not hasattr(self, "comparison_model_table"):
            return
        models = self._inference_local_models()
        self.comparison_model_table.setRowCount(len(models))
        for row, model_path in enumerate(models):
            selected_item = QTableWidgetItem()
            selected_item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            selected_item.setCheckState(Qt.CheckState.Unchecked)
            self.comparison_model_table.setItem(row, 0, selected_item)
            schema = inspect_model_schema(model_path)
            model_item = QTableWidgetItem(self._display_model_name(model_path))
            model_item.setData(Qt.ItemDataRole.UserRole, model_path)
            self.comparison_model_table.setItem(row, 1, model_item)
            values = [
                self._model_status(model_path),
                self._schema_signature(schema.labels),
            ]
            for column, value in enumerate(values, start=2):
                self.comparison_model_table.setItem(row, column, QTableWidgetItem(value))
        self.comparison_status_label.setText(
            "Выберите минимум две локальные обученные модели. Сравнение качества включится только для совместимых схем меток."
        )

    def _set_comparison_selection(self, checked: bool) -> None:
        if not hasattr(self, "comparison_model_table"):
            return
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in range(self.comparison_model_table.rowCount()):
            item = self.comparison_model_table.item(row, 0)
            if item is not None:
                item.setCheckState(state)

    def _refresh_comparison_inputs(self) -> None:
        if not hasattr(self, "comparison_dataset_combo"):
            return
        current_value = self.comparison_dataset_combo.currentData()
        options: list[tuple[str, str]] = []
        if not self.analysis_data_frame.empty:
            options.append(("analysis", "Файл анализа"))
        if not self.data_frame.empty:
            options.append(("train", "Train-датасет"))
        if not self.val_data_frame.empty:
            options.append(("validation", "Validation-датасет"))
        if not self.test_data_frame.empty:
            options.append(("test", "Test-датасет"))
        if not options:
            options = [("none", "Нет данных")]
        self.comparison_dataset_combo.blockSignals(True)
        self.comparison_dataset_combo.clear()
        for key, title in options:
            self.comparison_dataset_combo.addItem(title, key)
        target_index = next((index for index, (key, _) in enumerate(options) if key == current_value), 0)
        self.comparison_dataset_combo.setCurrentIndex(target_index)
        self.comparison_dataset_combo.blockSignals(False)
        self._populate_comparison_columns()
        self.populate_comparison_model_table()

    def _populate_comparison_columns(self) -> None:
        if not hasattr(self, "comparison_text_column_combo"):
            return
        frame = self._current_comparison_frame()
        self.comparison_text_column_combo.clear()
        self.comparison_label_column_combo.clear()
        self.comparison_label_column_combo.addItem("Без колонки метки")
        if frame.empty:
            self.comparison_text_column_combo.addItem("text")
            return
        columns = [str(column) for column in frame.columns]
        self.comparison_text_column_combo.addItems(columns)
        self.comparison_label_column_combo.addItems(columns)
        preferred_text = next(
            (column for column in columns if column.lower() in {"text", "текст", "review", "content", "message"}),
            columns[0],
        )
        preferred_label = next(
            (column for column in columns if column.lower() in {"label", "labels", "sentiment", "тональность", "class", "target", "y"}),
            "Без колонки метки",
        )
        self.comparison_text_column_combo.setCurrentText(preferred_text)
        if preferred_label in columns:
            self.comparison_label_column_combo.setCurrentText(preferred_label)

    def _current_comparison_frame(self) -> pd.DataFrame:
        if not hasattr(self, "comparison_dataset_combo"):
            return pd.DataFrame()
        mapping = {
            "analysis": self.analysis_data_frame,
            "train": self.data_frame,
            "validation": self.val_data_frame,
            "test": self.test_data_frame,
        }
        return mapping.get(str(self.comparison_dataset_combo.currentData()), pd.DataFrame())

    def _selected_comparison_models(self) -> list[str]:
        models: list[str] = []
        if not hasattr(self, "comparison_model_table"):
            return models
        for row in range(self.comparison_model_table.rowCount()):
            state_item = self.comparison_model_table.item(row, 0)
            model_item = self.comparison_model_table.item(row, 1)
            if state_item is None or model_item is None:
                continue
            if state_item.checkState() == Qt.CheckState.Checked:
                models.append(str(model_item.data(Qt.ItemDataRole.UserRole) or model_item.text()))
        return models

    def run_model_comparison(self) -> None:
        selected_models = self._selected_comparison_models()
        if len(selected_models) < 2:
            QMessageBox.information(self, "Недостаточно моделей", "Выберите минимум две локальные модели для сравнения.")
            return

        frame = self._current_comparison_frame()
        if frame.empty:
            QMessageBox.information(self, "Нет данных", "Загрузите датасет для анализа, train, validation или test.")
            return

        text_column = self.comparison_text_column_combo.currentText()
        label_column = self.comparison_label_column_combo.currentText()
        if text_column not in frame.columns:
            QMessageBox.warning(self, "Нет текстовой колонки", "Выберите корректную колонку с текстом.")
            return

        sources_column = self._guess_source_column(frame)
        records: list[tuple[str, str, str]] = []
        text_rows = 0
        use_labels = label_column and label_column != "Без колонки метки" and label_column in frame.columns
        for _, row in frame.iterrows():
            text = str(row.get(text_column, "")).strip()
            if not text:
                continue
            text_rows += 1
            label = ""
            if use_labels:
                if str(self.comparison_dataset_combo.currentData()) in {"train", "validation", "test"}:
                    mapped_label = self._map_label_value(row.get(label_column, ""))
                    label = str(mapped_label).strip() if mapped_label is not None else ""
                else:
                    label = str(row.get(label_column, "")).strip()
            if use_labels and not label:
                continue
            if use_labels:
                records.append((text, label, str(row.get(sources_column, "")).strip() if sources_column else ""))
            else:
                records.append((text, "", str(row.get(sources_column, "")).strip() if sources_column else ""))

        if not records:
            if use_labels and text_rows > 0:
                QMessageBox.warning(self, "Нет меток", "В выбранной колонке нет непустых истинных меток.")
                return
            QMessageBox.information(self, "Нет текстов", "В выбранной колонке нет непустых текстов.")
            return

        candidate_count = len(records)
        max_rows = self.comparison_max_rows_spin.value() if hasattr(self, "comparison_max_rows_spin") else 0
        selection_mode = str(self.comparison_sampling_combo.currentData() or "ordered")
        selection_note = f"все {format_int(candidate_count)}"
        if max_rows > 0 and candidate_count > max_rows:
            if selection_mode == "random":
                randomizer = random.Random(42)
                selected_indexes = sorted(randomizer.sample(range(candidate_count), max_rows))
                records = [records[index] for index in selected_indexes]
                selection_note = f"случайная подвыборка {format_int(max_rows)} из {format_int(candidate_count)}"
            else:
                records = records[:max_rows]
                selection_note = f"первые {format_int(max_rows)} из {format_int(candidate_count)}"

        texts = [text for text, _, _ in records]
        labels = [label for _, label, _ in records] if use_labels else []
        sources = [source for _, _, source in records]

        options = self._analysis_options()
        self._set_status("Сравнение моделей", "running")
        self.statusBar().showMessage("Выполняется сравнение моделей...")
        QApplication.processEvents()

        def comparison_progress(model_name: str, processed: int, total: int) -> None:
            short_name = self._display_model_name(model_name)
            self.statusBar().showMessage(
                f"Сравнение моделей... {short_name}: {format_int(processed)} / {format_int(total)}"
            )
            QApplication.processEvents()

        try:
            result = compare_models(
                selected_models,
                texts,
                options,
                self._analysis_device(),
                true_labels=labels or None,
                sources=sources,
                batch_size=None,
                progress_callback=comparison_progress,
            )
        except TransformerLoadError as exc:
            self._set_status("Ошибка", "error")
            self.statusBar().showMessage("Ошибка сравнения моделей.")
            QMessageBox.critical(self, "Ошибка сравнения моделей", str(exc))
            return

        self.comparison_behavior_rows = result.behavior_rows
        self.comparison_quality_rows = result.quality_rows
        self._populate_comparison_quality_table(result)
        self._populate_comparison_behavior_table(result)
        self._populate_comparison_disagreement_table(result)
        self._populate_comparison_charts(result)
        self.comparison_quality_summary.setText(self._build_comparison_quality_summary(result))
        self.comparison_status_label.setText(
            f"Сравнение завершено: {format_int(len(selected_models))} моделей, {format_int(len(texts))} текстов "
            f"({selection_note}), расхождений найдено {format_int(len(result.disagreements))}."
        )
        self._set_status("Сравнение выполнено", "ready")
        self.statusBar().showMessage("Сравнение моделей завершено.")
        self._log(
            f"Сравнение моделей завершено: {len(selected_models)} моделей, {len(texts)} текстов "
            f"({selection_note}), {len(result.disagreements)} расхождений."
        )

    def _populate_comparison_quality_table(self, result: object) -> None:
        rows = getattr(result, "quality_rows", [])
        self.comparison_quality_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            values = [
                self._display_model_name(item.model_name),
                item.schema_signature,
                format_int(item.sample_count),
                f"{item.accuracy:.3f}",
                f"{item.macro_f1:.3f}",
            ]
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                if column == 0:
                    table_item.setToolTip(item.model_name)
                self.comparison_quality_table.setItem(row, column, table_item)

    def _populate_comparison_behavior_table(self, result: object) -> None:
        rows = getattr(result, "behavior_rows", [])
        self.comparison_behavior_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            values = [
                self._display_model_name(item.model_name),
                self._schema_signature(item.schema.labels),
                f"{item.avg_confidence:.3f}",
                f"{item.median_confidence:.3f}",
                f"{item.low_confidence_rate:.1%}",
                f"{item.prediction_entropy:.3f}",
                f"{item.top_class_share:.1%}",
                f"{item.inference_seconds:.2f}",
            ]
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                if column == 0:
                    table_item.setToolTip(item.model_name)
                self.comparison_behavior_table.setItem(row, column, table_item)

    def _populate_comparison_disagreement_table(self, result: object) -> None:
        rows = getattr(result, "disagreements", [])
        self.comparison_disagreement_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            predictions = " | ".join(
                f"{self._display_model_name(model)}: {label} ({item.confidences.get(model, 0.0):.2f})"
                for model, label in item.predictions.items()
            )
            values = [
                str(row + 1),
                item.text[:260],
                predictions,
            ]
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                if column == 2:
                    table_item.setToolTip(predictions)
                self.comparison_disagreement_table.setItem(row, column, table_item)

    def _populate_comparison_charts(self, result: object) -> None:
        quality_rows = getattr(result, "quality_rows", [])
        behavior_rows = getattr(result, "behavior_rows", [])

        if quality_rows:
            quality_labels = [self._display_model_name(item.model_name) for item in quality_rows]
            self.comparison_quality_chart.draw_grouped_bars(
                quality_labels,
                [
                    ("Accuracy", [item.accuracy for item in quality_rows], "#7aa5dc"),
                    ("Macro F1", [item.macro_f1 for item in quality_rows], "#d97706"),
                ],
                y_max=1.0,
                rotate_labels=20,
            )
        else:
            self.comparison_quality_chart.empty("Нет метрик качества")

        if behavior_rows:
            behavior_labels = [self._display_model_name(item.model_name) for item in behavior_rows]
            self.comparison_behavior_chart.draw_grouped_bars(
                behavior_labels,
                [
                    ("Средняя уверенность", [item.avg_confidence for item in behavior_rows], "#7aa5dc"),
                    ("Низкая уверенность", [item.low_confidence_rate for item in behavior_rows], "#ef4444"),
                ],
                y_max=1.0,
                rotate_labels=20,
            )
            self.comparison_speed_chart.draw_grouped_bars(
                behavior_labels,
                [("Секунды", [item.inference_seconds for item in behavior_rows], "#16a34a")],
                rotate_labels=20,
            )
            self._draw_comparison_disagreement_chart(behavior_rows)
        else:
            self.comparison_behavior_chart.empty("Нет данных поведения")
            self.comparison_speed_chart.empty("Нет данных по времени")
            self.comparison_disagreement_chart.empty("Нет данных о расхождениях")

    def _draw_comparison_disagreement_chart(self, rows: list[object]) -> None:
        if len(rows) < 2:
            self.comparison_disagreement_chart.empty("Нужно минимум 2 модели")
            return

        labels = [self._display_model_name(getattr(row, "model_name", "")) for row in rows]
        sizes = [len(getattr(row, "predictions", [])) for row in rows]
        total = min(sizes) if sizes else 0
        if total <= 0:
            self.comparison_disagreement_chart.empty("Нет предсказаний")
            return

        matrix: list[list[float]] = []
        for row_index, left_row in enumerate(rows):
            left_predictions = getattr(left_row, "predictions", [])
            row_values: list[float] = []
            for column_index, right_row in enumerate(rows):
                if row_index == column_index:
                    row_values.append(0.0)
                    continue
                right_predictions = getattr(right_row, "predictions", [])
                disagreements = 0
                for index in range(total):
                    if left_predictions[index].sentiment != right_predictions[index].sentiment:
                        disagreements += 1
                row_values.append(disagreements / total if total else 0.0)
            matrix.append(row_values)
        self.comparison_disagreement_chart.draw_heatmap(labels, matrix)

    def _format_comparison_notes(self, notes: list[str]) -> str:
        lines: list[str] = []
        for note in notes:
            compact = note
            for model_name in self._selected_comparison_models():
                compact = compact.replace(model_name, self._display_model_name(model_name))
            if compact:
                lines.append(compact)
        return "\n".join(lines)

    def _build_comparison_quality_summary(self, result: object) -> str:
        quality_rows = getattr(result, "quality_rows", [])
        quality_groups = getattr(result, "quality_groups", {})
        notes = self._format_comparison_notes(getattr(result, "quality_notes", []))

        lines: list[str] = []
        if quality_rows:
            lines.append(f"Метрики качества посчитаны для {format_int(len(quality_rows))} моделей.")
        else:
            lines.append("Метрики качества пока недоступны.")

        compatible = [
            f"{signature} - {format_int(len(models))} мод."
            for signature, models in quality_groups.items()
            if len(models) >= 2
        ]
        if compatible:
            lines.append("Совместимые схемы: " + "; ".join(compatible) + ".")

        if notes:
            lines.append("Причины:")
            lines.extend(f"• {line}" for line in notes.splitlines() if line.strip())
        return "\n".join(lines)

    @staticmethod
    def _format_profile_metric(value: object) -> str:
        if value in (None, "", "-"):
            return "-"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _schema_signature(labels: tuple[str, ...] | list[str]) -> str:
        if not labels:
            return "неизвестно"
        text = " | ".join(str(label) for label in labels)
        return text if len(text) <= 120 else f"{text[:117]}..."

    @staticmethod
    def _model_identity(model_name: str) -> str:
        path = Path(model_name)
        if path.exists():
            return str(path.resolve()).casefold()
        return model_name.casefold()

    def _existing_local_models(self) -> list[str]:
        models_dir = Path("models")
        if not models_dir.exists():
            return []
        paths: dict[str, str] = {}
        for config_path in models_dir.rglob("config.json"):
            folder = config_path.parent
            if not folder.is_dir():
                continue
            if not self._looks_like_local_model(folder):
                continue
            resolved = str(folder.resolve())
            paths[resolved.casefold()] = resolved
        return sorted(paths.values(), key=lambda value: Path(value).name.casefold())

    def _inference_local_models(self) -> list[str]:
        return [model_path for model_path in self._existing_local_models() if self._is_trained_inference_model(Path(model_path))]

    @staticmethod
    def _looks_like_local_model(folder: Path) -> bool:
        has_weights = any((folder / name).exists() for name in WEIGHT_FILES)
        has_tokenizer = any((folder / name).exists() for name in TOKENIZER_VOCAB_FILES)
        return has_weights and has_tokenizer

    @staticmethod
    def _is_trained_inference_model(folder: Path) -> bool:
        return (folder / "training_metrics.json").exists()

    def _local_model_profile(self, model_path: str) -> dict[str, float | int | str | bool]:
        path = Path(model_path)
        metrics_path = path / "training_metrics.json"
        metrics: dict[str, object] = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metrics = {}
        is_trained = bool(metrics)
        display_name = path.name if is_trained else self._training_base_display_name(path)
        schema = inspect_model_schema(model_path)
        return {
            "name": str(path.resolve()) if path.exists() else model_path,
            "display_name": display_name,
            "status": "локальная обученная" if metrics else self._model_status(model_path),
            "label_schema": self._schema_signature(schema.labels),
            "quality_group": schema.signature if schema.labels else "-",
            "accuracy": metrics.get("accuracy", "-"),
            "macro_f1": metrics.get("macro_f1", "-"),
            "can_rename": is_trained,
            "registry_key": str(path.resolve()).casefold() if is_trained else f"base::{display_name.casefold()}",
        }

    @staticmethod
    def _model_config_dict(model_path: Path) -> dict[str, object]:
        config_path = model_path / "config.json"
        if not config_path.exists():
            return {}
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _training_base_display_name(self, model_path: Path) -> str:
        config = self._model_config_dict(model_path)
        raw_name = str(config.get("_name_or_path") or config.get("name_or_path") or "").strip()
        if raw_name and raw_name not in {".", ".."} and not raw_name.startswith(str(model_path)):
            return Path(raw_name).name if raw_name.endswith(tuple(WEIGHT_FILES)) else raw_name

        for part in model_path.parts:
            if part.startswith("models--"):
                repo_name = part.removeprefix("models--").replace("--", "/").strip("/")
                if repo_name:
                    return repo_name
        return model_path.name

    def _find_profile_row(self, model_name: str) -> int | None:
        if not hasattr(self, "profile_table"):
            return None
        target = self._model_identity(model_name)
        for row in range(self.profile_table.rowCount()):
            item = self.profile_table.item(row, 0)
            if item is None:
                continue
            current = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if self._model_identity(current) == target:
                return row
        return None

    def _handle_profile_item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating_profile_table or item.column() != 0:
            return
        source_name = str(item.data(Qt.ItemDataRole.UserRole) or "")
        pending_name = self._pending_profile_rename_path or ""
        if not source_name or self._model_identity(source_name) != self._model_identity(pending_name):
            return

        path = Path(source_name)
        new_name = item.text().strip()
        old_name = self._pending_profile_rename_original_name or path.name
        self._pending_profile_rename_path = None
        self._pending_profile_rename_original_name = ""

        if not new_name or new_name == old_name:
            self.populate_profile_table()
            return
        if any(char in new_name for char in '<>:"/\\|?*'):
            QMessageBox.warning(self, "Недопустимое имя", "Имя папки содержит недопустимые символы.")
            self.populate_profile_table()
            return

        target_path = path.with_name(new_name)
        if target_path.exists():
            QMessageBox.warning(self, "Имя занято", "Папка с таким именем уже существует.")
            self.populate_profile_table()
            return

        try:
            path.rename(target_path)
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось переименовать", str(exc))
            self.populate_profile_table()
            return

        self._refresh_model_lists(str(target_path.resolve()))
        self.statusBar().showMessage(f"Модель переименована: {target_path.name}")
        self._log(f"Модель переименована: {path.name} -> {target_path.name}.")

    def apply_filter(self, text: str) -> None:
        if not text:
            self.populate_results_table(self.results)
            return
        query = text.casefold()
        filtered = [
            result
            for result in self.results
            if query in result.text.casefold() or query in result.sentiment.casefold()
        ]
        self.populate_results_table(filtered)

    def _select_model_from_profile(self, row: int, column: int) -> None:
        if self.profile_table.item(row, 0) is not None:
            item = self.profile_table.item(row, 0)
            model_name = str(item.data(Qt.ItemDataRole.UserRole) or item.text())
            if not self._is_trained_inference_model(Path(model_name)):
                self.statusBar().showMessage("Это базовая модель для обучения, не готовая модель анализа.")
                return
            self.model_combo.setCurrentText(model_name)
            self.statusBar().showMessage("Модель анализа выбрана из профиля.")

    def _read_dataset(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return self._read_csv(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix == ".txt":
            lines = self._read_text(path).splitlines()
            return pd.DataFrame({"text": [line for line in lines if line.strip()]})
        raise ValueError(f"Неподдерживаемый формат файла: {suffix}")

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Не удалось прочитать CSV: {last_error}") from last_error

    @staticmethod
    def _read_text(path: Path) -> str:
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        raise ValueError(f"Не удалось прочитать TXT: {last_error}") from last_error

    def _populate_analysis_columns(self) -> None:
        self.text_column_combo.clear()
        columns = [str(column) for column in self.analysis_data_frame.columns]
        preferred = next(
            (column for column in columns if column.lower() in {"text", "текст", "review", "content", "message"}),
            columns[0] if columns else "text",
        )
        self.text_column_combo.addItems(columns)
        self.text_column_combo.setCurrentText(preferred)
        self._log(f"Колонка текста для анализа: {preferred}.")

    def _populate_training_columns(self) -> None:
        columns = [str(column) for column in self.data_frame.columns]
        text_preferred = self.text_column_combo.currentText()
        label_preferred = next(
            (column for column in columns if column.lower() in {"label", "labels", "sentiment", "тональность", "class", "target", "y"}),
            columns[1] if len(columns) > 1 else (columns[0] if columns else ""),
        )

        self.train_text_column_combo.clear()
        self.train_label_column_combo.blockSignals(True)
        self.train_label_column_combo.clear()
        self.train_text_column_combo.addItems(columns)
        self.train_label_column_combo.addItems(columns)
        if text_preferred in columns:
            self.train_text_column_combo.setCurrentText(text_preferred)
        if label_preferred in columns:
            self.train_label_column_combo.setCurrentText(label_preferred)
        self.train_label_column_combo.blockSignals(False)

    def _guess_source_column(self, frame: pd.DataFrame | None = None) -> str | None:
        source = self.analysis_data_frame if frame is None else frame
        for column in source.columns:
            if str(column).lower() in {"source", "источник", "platform", "site"}:
                return str(column)
        return None

    @staticmethod
    def _model_status(name: str) -> str:
        path = Path(name)
        if path.exists():
            if (path / "training_metrics.json").exists():
                return "локальная обученная"
            return "базовая для обучения"
        if name.startswith("models/"):
            return "не найдена"
        return "Профиль Hugging Face"

    def _refresh_empty_charts(self) -> None:
        self.class_chart.empty()
        self.confidence_chart.empty()
        self.drift_chart.empty()
        self.refresh_monitoring()

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        self.event_log.append((timestamp, message))
        self.statusBar().showMessage(message)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 12px;
                color: #172033;
            }
            QMainWindow {
                background: #eef2f7;
            }
            QStatusBar {
                background: #f8fafc;
                border-top: 1px solid rgba(129, 145, 166, 0.35);
                color: #5b677a;
            }
            QLabel#statusMessageLabel {
                color: #5b677a;
                padding: 0 12px;
            }
            #contextBar {
                background: #f8fafc;
                border-bottom: 1px solid rgba(129, 145, 166, 0.35);
                border-radius: 0;
            }
            #contextSeparator {
                color: rgba(129, 145, 166, 0.5);
                font-size: 13px;
                padding: 0 2px;
            }
            #contextTitle {
                color: #7b8798;
                font-size: 11px;
                font-weight: 400;
            }
            #contextValue {
                min-height: 28px;
                padding: 4px 8px;
                border-radius: 4px;
                border: 1px solid rgba(129, 145, 166, 0.32);
                background: #eef2f7;
                color: #1d2a3d;
                font-weight: 400;
            }
            #contextSmallValue {
                color: #5b677a;
                font-weight: 400;
            }
            #contextCombo {
                min-height: 28px;
                border: 1px solid rgba(107, 120, 140, 0.58);
                border-radius: 4px;
                background: #ffffff;
                padding: 4px 8px;
            }
            #contextCombo:hover {
                border-color: #6e9bd6;
            }
            #contextCombo:focus {
                border-color: #256fc7;
            }
            #contextButton {
                min-height: 28px;
                padding: 4px 12px;
                border-radius: 4px;
                border: 1px solid rgba(107, 120, 140, 0.58);
                background: #f8fafc;
                color: #1f2937;
                font-weight: 400;
            }
            #contextButton:hover {
                background: #eef5ff;
                border-color: #6e9bd6;
            }
            #contextButton:focus {
                border-color: #256fc7;
            }
            #contextText {
                color: #5b677a;
                font-weight: 400;
            }
            #fieldHelp {
                color: #6b7280;
                font-size: 11px;
            }
            #sidebar {
                background: #eef2f7;
                border-right: 1px solid rgba(129, 145, 166, 0.35);
                border-radius: 0;
            }
            #appTitle {
                font-size: 13px;
                font-weight: 400;
                color: #1d2a3d;
            }
            #sidebarSection {
                color: #7b8798;
                font-size: 11px;
                font-weight: 400;
                padding-top: 4px;
                letter-spacing: 0.4px;
            }
            #formLabel {
                color: #526070;
                font-size: 11px;
                font-weight: 400;
            }
            #pageHeader {
                background: #f8fafc;
                border: 1px solid rgba(129, 145, 166, 0.32);
                border-radius: 6px;
            }
            #pageTitle {
                color: #111827;
                font-size: 18px;
                font-weight: 600;
            }
            #pageSubtitle {
                color: #5b677a;
                font-size: 12px;
            }
            #panel, #metricCard {
                background: #ffffff;
                border: 1px solid rgba(129, 145, 166, 0.38);
                border-radius: 6px;
            }
            #workbenchPanel {
                background: #ffffff;
                border: 1px solid rgba(129, 145, 166, 0.38);
                border-radius: 6px;
            }
            #panelTitle {
                font-size: 13px;
                font-weight: 600;
                color: #111827;
            }
            #metricTitle {
                color: #5b677a;
                font-size: 11px;
                font-weight: 400;
            }
            #metricValue {
                color: #111827;
                font-size: 22px;
                font-weight: 400;
            }
            #mutedLabel {
                color: #6b7280;
            }
            QFrame#comparisonToolbar {
                background: #f8fafc;
                border: 1px solid rgba(129, 145, 166, 0.28);
                border-radius: 6px;
            }
            #comparisonFieldLabel {
                color: #526070;
                font-size: 11px;
                font-weight: 400;
                margin-bottom: 2px;
            }
            #comparisonHint {
                color: #6b7280;
                font-size: 11px;
            }
            #comparisonStatusBox {
                background: #fbfcfe;
                border: 1px solid rgba(129, 145, 166, 0.28);
                border-radius: 6px;
                color: #344054;
                padding: 8px 10px;
            }
            #mappingSummary {
                background: #f8fafc;
                border: 1px solid rgba(129, 145, 166, 0.32);
                border-radius: 4px;
                color: #344054;
                padding: 8px;
                font-weight: 400;
            }
            #analysisSettingsSection {
                background: #fbfcfe;
                border: 1px solid rgba(129, 145, 166, 0.24);
                border-radius: 4px;
            }
            QFrame#softSeparator {
                background: rgba(129, 145, 166, 0.22);
                border: 0;
            }
            #quickResultPanel {
                background: #fbfcfe;
                border: 1px solid rgba(129, 145, 166, 0.38);
                border-radius: 4px;
            }
            QPlainTextEdit#quickTextInput {
                min-height: 240px;
            }
            QTableWidget#quickProbabilityTable {
                background: #ffffff;
                alternate-background-color: #f7f9fc;
                gridline-color: #e5eaf1;
                border: 1px solid rgba(129, 145, 166, 0.32);
                border-radius: 4px;
                selection-background-color: #dbeafe;
                selection-color: #111827;
            }
            QTableWidget#quickProbabilityTable::item {
                padding: 4px;
            }
            QTextBrowser#reportPreview {
                border: 1px solid rgba(129, 145, 166, 0.38);
                border-radius: 4px;
                background: #ffffff;
                padding: 12px;
            }
            #quickResult {
                color: #111827;
                font-size: 16px;
                font-weight: 400;
            }
            #quickResult[sentiment="positive"] { color: #2563eb; }
            #quickResult[sentiment="neutral"] { color: #d97706; }
            #quickResult[sentiment="negative"] { color: #dc2626; }
            QPushButton {
                min-height: 28px;
                padding: 4px 12px;
                border-radius: 4px;
                border: 1px solid rgba(107, 120, 140, 0.58);
                background: #f8fafc;
                color: #1f2937;
                font-weight: 400;
            }
            QPushButton:hover {
                background: #eef5ff;
                border-color: #6e9bd6;
            }
            QPushButton:pressed {
                background: #e1ecfb;
            }
            QPushButton:disabled {
                color: #9ca3af;
                background: #f1f4f8;
                border-color: rgba(107, 120, 140, 0.28);
            }
            QPushButton#primaryButton {
                background: #256fc7;
                border-color: #1f5fa9;
                color: white;
                font-weight: 600;
            }
            QPushButton#primaryButton:hover { background: #1f65b8; }
            QFrame#analysisMetricsStrip {
                background: #fbfcfe;
                border: 1px solid rgba(129, 145, 166, 0.28);
                border-radius: 4px;
                padding: 5px 8px;
            }
            QLabel#analysisMetricText {
                color: #526070;
                font-size: 11px;
            }
            QPushButton#dangerButton {
                background: #fff1f1;
                border-color: rgba(220, 38, 38, 0.42);
                color: #991b1b;
            }
            QPushButton#dangerButton:hover {
                background: #fee2e2;
                border-color: rgba(220, 38, 38, 0.62);
            }
            QPushButton#dangerButton:disabled {
                color: #9ca3af;
                background: #f1f4f8;
                border-color: rgba(107, 120, 140, 0.28);
            }
            QPushButton#tableActionButton {
                padding: 0px 12px;
            }
            QPushButton#tableDangerActionButton {
                padding: 0px 12px;
                background: #fff1f1;
                border-color: rgba(220, 38, 38, 0.42);
                color: #991b1b;
            }
            QPushButton#tableDangerActionButton:hover {
                background: #fee2e2;
                border-color: rgba(220, 38, 38, 0.62);
            }
            QPushButton#tableDangerActionButton:disabled {
                color: #9ca3af;
                background: #f1f4f8;
                border-color: rgba(107, 120, 140, 0.28);
            }
            QPushButton#modeButton {
                min-height: 28px;
                padding: 4px 12px;
                border-radius: 4px;
                border: 1px solid rgba(107, 120, 140, 0.42);
                background: #f8fafc;
                color: #344054;
            }
            QPushButton#modeButton:checked {
                background: #dbeafe;
                border-color: rgba(77, 124, 191, 0.42);
                color: #1f5fa9;
                font-weight: 400;
            }
            QPushButton#navButton {
                min-height: 28px;
                text-align: left;
                padding-left: 8px;
                border: 1px solid transparent;
                background: transparent;
                color: #344054;
            }
            QPushButton#navButton:hover {
                background: rgba(215, 229, 251, 0.58);
            }
            QPushButton#navButton:checked {
                background: #dbeafe;
                border-color: rgba(77, 124, 191, 0.42);
                color: #1f5fa9;
                font-weight: 400;
            }
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
                min-height: 28px;
                border: 1px solid rgba(107, 120, 140, 0.58);
                border-radius: 4px;
                background: #ffffff;
                padding: 4px 8px;
            }
            QComboBox#tableCombo {
                min-height: 32px;
                max-height: 32px;
                margin: 0px;
                padding: 0px 8px;
            }
            QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #6e9bd6;
            }
            QPlainTextEdit {
                border: 1px solid rgba(129, 145, 166, 0.38);
                border-radius: 4px;
                background: #fbfcfe;
                color: #172033;
                padding: 8px;
                selection-background-color: #dbeafe;
            }
            QPlainTextEdit#trainingLog {
                border: 0;
                border-radius: 0;
                font-family: Consolas, "Cascadia Mono", monospace;
                font-size: 12px;
                line-height: 1.35;
            }
            QStackedWidget#trainingLogStack {
                border: 1px solid rgba(129, 145, 166, 0.38);
                border-radius: 4px;
                background: #fbfcfe;
            }
            QFrame#trainingLogEmpty {
                background: #fbfcfe;
                border: 0;
            }
            QLabel#emptyLogMessage {
                color: #6b7280;
                font-size: 14px;
            }
            QCheckBox {
                spacing: 8px;
                min-height: 24px;
                color: #344054;
            }
            QTableWidget#embeddedTable {
                background: #fbfcfe;
                alternate-background-color: #f7f9fc;
                gridline-color: #e5eaf1;
                border: 0;
                border-radius: 0;
                selection-background-color: #dbeafe;
                selection-color: #111827;
            }
            QTableWidget#embeddedTable::item {
                padding: 4px;
            }
            QHeaderView::section {
                background: #f3f6fa;
                color: #526070;
                border: 0;
                border-bottom: 1px solid rgba(129, 145, 166, 0.32);
                border-right: 1px solid rgba(129, 145, 166, 0.32);
                padding: 6px;
                font-weight: 400;
                font-size: 11px;
            }
            QHeaderView::section:last {
                border-right: 0;
            }
            QProgressBar {
                min-height: 12px;
                border: 1px solid rgba(129, 145, 166, 0.38);
                border-radius: 4px;
                background: #f1f4f8;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 4px;
                background: #256fc7;
            }
            QSplitter::handle {
                background: transparent;
            }
            QSplitter::handle:horizontal {
                width: 4px;
            }
            #trainingTabs {
                background: #fbfcfe;
            }
            #trainingTabStack {
                border: 1px solid rgba(129, 145, 166, 0.32);
                border-radius: 4px;
                background: #fbfcfe;
            }
            QPushButton#trainingTabButton {
                min-height: 32px;
                padding: 4px 8px;
                border: 1px solid rgba(107, 120, 140, 0.42);
                border-bottom: 0;
                border-radius: 0;
                background: #f8fafc;
                color: #344054;
                font-size: 12px;
                font-weight: 400;
            }
            QPushButton#trainingTabButton:hover {
                background: #eef5ff;
                border-color: #6e9bd6;
            }
            QPushButton#trainingTabButton:checked {
                background: #dbeafe;
                color: #1f5fa9;
                border-color: rgba(77, 124, 191, 0.42);
            }
            QScrollArea#trainingScroll {
                background: #fbfcfe;
                border: 0;
            }
            QLabel#statusChip {
                min-height: 28px;
                border-radius: 6px;
                padding: 4px 12px;
                font-size: 12px;
                font-weight: 400;
            }
            QLabel#statusChip[state="ready"] {
                background: #f0f8f2;
                border: 1px solid rgba(34, 126, 62, 0.30);
                color: #166534;
            }
            QLabel#statusChip[state="running"] {
                background: #eef5ff;
                border: 1px solid rgba(37, 111, 199, 0.30);
                color: #1f5fa9;
            }
            QLabel#statusChip[state="error"] {
                background: #fff1f1;
                border: 1px solid rgba(220, 38, 38, 0.30);
                color: #991b1b;
            }
            QStatusBar[state="ready"] {
                color: #166534;
            }
            QStatusBar[state="running"] {
                color: #1f5fa9;
            }
            QStatusBar[state="error"] {
                color: #991b1b;
            }
            QLabel#sidebarHint {
                color: #7b8798;
                font-size: 11px;
                padding: 8px 4px;
                line-height: 1.4;
            }
            #emptyStateKicker {
                color: #256fc7;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.8px;
            }
            #emptyStateTitle {
                color: #111827;
                font-size: 18px;
                font-weight: 600;
            }
            #emptyStateMessage {
                color: #526070;
                font-size: 12px;
                line-height: 1.5;
            }
            #emptyStateStepText {
                color: #5b677a;
                font-size: 12px;
                line-height: 1.5;
            }
            """
        )
