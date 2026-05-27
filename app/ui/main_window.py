"""Main PyQt window for the sentiment laboratory.

Версия с исправлениями по layout и подготовке меток:
- верхняя панель используется как панель рабочего контекста, а не как набор дублирующих действий;
- кнопка загрузки явно называется «Загрузить данные»;
- параметры анализа перенесены в область анализа, а не смешаны с навигацией;
- добавлена настройка меток для обучения: исключение лишних классов, маппинг к 3 классам,
  поддержка строк со списками меток вроде "(3,5,6)";
- перед обучением формируется очищенный набор text/target_label;
- таблицы и правые панели рассчитаны на нормальное масштабирование окна.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.models.sentiment import (
    AnalysisResult,
    MODEL_PROFILES,
    TRAINING_BASE_MODELS,
    SentimentAnalyzer,
    TransformerLoadError,
)
from app.monitoring import build_drift_report
from app.preprocessing import PreprocessingOptions
from app.reports import export_html_report
from app.training.experiments import class_distribution, comparison_rows, summarize_results
from app.training.transformer_trainer import TrainConfig, TrainResult, TrainingError, train_transformer_classifier


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
    "Колонка текста": "Столбец датасета, из которого берутся тексты для fine-tuning.",
    "Колонка метки": "Столбец с исходными метками; лишние значения настраиваются во вкладке Метки.",
    "Максимум строк": "Ограничивает размер обучающего набора для быстрых пробных запусков; 0 значит без ограничения.",
    "Очистка": "Удаляет дубли и перемешивает строки перед разбиением на train/validation.",
    "Validation split": "Доля данных, отложенная для проверки качества после каждой эпохи.",
    "Split seed": "Отдельный seed для разбиения данных на train и validation.",
    "Стратегия": "Стратификация сохраняет похожее распределение классов в train и validation.",
    "Веса классов": "Balanced повышает вклад редких классов, если данные несбалансированы.",
    "Лучшая модель по": "Метрика, по которой выбирается лучший checkpoint и работает early stopping.",
    "Базовая модель": "Pretrained checkpoint, от которого начинается дообучение классификатора.",
    "Тип задачи": "Для текущего sentiment-classifier используется single-label classification.",
    "Загрузка": "Контролирует локальную загрузку, trust_remote_code и замену несовпадающей головы классификации.",
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
    "Сохранение": "Сохраняет параметры запуска и predictions на validation для анализа ошибок.",
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


def make_combo(values: list[str], current: str) -> QComboBox:
    combo = QComboBox()
    combo.addItems(values)
    if current in values:
        combo.setCurrentText(current)
    return combo


class Panel(QFrame):
    def __init__(self, title: str = "") -> None:
        super().__init__()
        self.setObjectName("panel")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(SPACE_3, SPACE_2, SPACE_3, SPACE_3)
        self.layout.setSpacing(SPACE_2)
        if title:
            label = QLabel(title)
            label.setObjectName("panelTitle")
            self.layout.addWidget(label)


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "0", subtitle: str = "") -> None:
        super().__init__()
        self.setObjectName("metricCard")
        self.setMinimumHeight(72)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE_3, SPACE_2, SPACE_3, SPACE_2)
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


class TrainingThread(QThread):
    message = pyqtSignal(str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, texts: list[str], labels: list[object], config: TrainConfig) -> None:
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.config = config

    def run(self) -> None:
        try:
            result = train_transformer_classifier(self.texts, self.labels, self.config, self.message.emit)
        except TrainingError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover
            self.failed.emit(f"Ошибка обучения: {exc}")
        else:
            self.completed.emit(result)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Лаборатория анализа тональности")
        self.resize(1280, 820)
        self.setMinimumSize(900, 580)

        self.dataset_path: Path | None = None
        self.data_frame = pd.DataFrame()
        self.results: list[AnalysisResult] = []
        self.event_log: list[tuple[str, str]] = []
        self.preview_offset = 0
        self.nav_buttons: list[QPushButton] = []
        self.training_thread: TrainingThread | None = None
        self.label_action_combos: dict[str, QComboBox] = {}
        self.label_count_by_token: Counter[str] = Counter()

        self._build_ui()
        self._apply_styles()
        self._set_initial_state()

    def _build_ui(self) -> None:
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

    def _build_context_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("contextBar")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(SPACE_3, SPACE_2, SPACE_3, SPACE_2)
        layout.setSpacing(SPACE_2)

        self.load_data_button = QPushButton("Загрузить датасет")
        self.load_data_button.setObjectName("contextButton")
        self.load_data_button.setToolTip("Открыть CSV, TXT или Excel-файл с текстовыми данными")
        self.load_data_button.clicked.connect(self.load_dataset)

        dataset_title = QLabel("Файл:")
        dataset_title.setObjectName("contextTitle")

        self.dataset_name_label = QLabel("не загружен")
        self.dataset_name_label.setObjectName("contextValue")
        self.dataset_name_label.setMinimumWidth(180)
        self.dataset_name_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.dataset_rows_label = QLabel("Строк: 0")
        self.dataset_rows_label.setObjectName("contextSmallValue")

        device_title = QLabel("Устройство:")
        device_title.setObjectName("contextTitle")

        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "CUDA (0)", "Auto"])
        self.device_combo.setObjectName("contextCombo")
        self.device_combo.setFixedWidth(110)

        layout.addWidget(self.load_data_button)
        layout.addSpacing(4)

        layout.addWidget(dataset_title)
        layout.addWidget(self.dataset_name_label, 1)
        layout.addWidget(self.dataset_rows_label)

        layout.addSpacing(12)
        layout.addWidget(device_title)
        layout.addWidget(self.device_combo)

        return bar

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
        nav = ["Данные", "Анализ", "Обучение", "Модели", "Мониторинг", "Отчеты"]
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
        self.status_label = QLabel("Готово")
        self.status_label.setObjectName("statusChip")
        self.status_label.setProperty("state", "ready")
        layout.addWidget(self.status_label)
        return side

    def _build_data_page(self) -> QWidget:
        page = self._page()
        summary = Panel("Данные")
        row = QHBoxLayout()
        open_button = QPushButton("Загрузить датасет")
        open_button.clicked.connect(self.load_dataset)
        row.addWidget(open_button)
        self.data_summary_label = QLabel("Файл не открыт. Поддерживаются CSV, TXT, XLSX/XLS.")
        self.data_summary_label.setObjectName("mutedLabel")
        row.addWidget(self.data_summary_label, 1)
        summary.layout.addLayout(row)
        page.layout().addWidget(summary)

        data_metrics = QGridLayout()
        data_metrics.setHorizontalSpacing(SPACE_3)
        data_metrics.setVerticalSpacing(SPACE_3)
        self.dataset_rows_card = MetricCard("Строки", "0", "ожидает файл")
        self.dataset_columns_card = MetricCard("Колонки", "0", "структура набора")
        self.dataset_preview_card = MetricCard("Предпросмотр", "0", "строк в таблице")
        for column, card in enumerate((self.dataset_rows_card, self.dataset_columns_card, self.dataset_preview_card)):
            data_metrics.addWidget(card, 0, column)
            data_metrics.setColumnStretch(column, 1)
        page.layout().addLayout(data_metrics)

        preview = Panel("Предпросмотр")
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
        preview.layout.addWidget(self.preview_table)
        page.layout().addWidget(preview, 1)
        return page

    def _build_analysis_page(self) -> QWidget:
        page = self._page()

        settings = Panel("Параметры анализа")
        settings_form = QVBoxLayout()
        settings_form.setContentsMargins(0, 0, 0, 0)
        settings_form.setSpacing(SPACE_2)

        source_row = QHBoxLayout()
        source_row.setSpacing(SPACE_2)

        self.text_column_combo = QComboBox()
        self.text_column_combo.addItem("text")
        self.text_column_combo.setFixedWidth(176)

        self.model_combo = QComboBox()
        self.model_combo.addItems([str(profile["name"]) for profile in MODEL_PROFILES])
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
        self.analyze_button.clicked.connect(self.run_analysis)

        source_row.addWidget(QLabel("Текст:"))
        source_row.addWidget(self.text_column_combo)
        source_row.addSpacing(SPACE_3)
        source_row.addWidget(QLabel("Модель:"))
        source_row.addWidget(self.model_combo, 1)
        settings_form.addLayout(source_row)

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
        settings.layout.addWidget(hint)
        page.layout().addWidget(settings)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(SPACE_1)
        self.analysis_mode_group = QButtonGroup(self)
        self.analysis_mode_group.setExclusive(True)
        batch_button = QPushButton("Пакетный анализ")
        batch_button.setObjectName("modeButton")
        batch_button.setCheckable(True)
        batch_button.setChecked(True)
        quick_button = QPushButton("Один текст")
        quick_button.setObjectName("modeButton")
        quick_button.setCheckable(True)
        for index, button in enumerate((batch_button, quick_button)):
            self.analysis_mode_group.addButton(button, index)
            button.clicked.connect(lambda checked=False, page_index=index: self._switch_analysis_mode(page_index))
            mode_row.addWidget(button)
        mode_row.addStretch(1)
        page.layout().addLayout(mode_row)

        self.analysis_mode_stack = QStackedWidget()
        batch_page = QWidget()
        batch_layout = QVBoxLayout(batch_page)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.setSpacing(SPACE_3)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(SPACE_3)
        metrics.setVerticalSpacing(SPACE_3)
        self.total_card = MetricCard("Всего текстов", "0", "в наборе данных")
        self.processed_card = MetricCard("Проанализировано", "0", "строк")
        self.confidence_card = MetricCard("Средняя уверенность", "0.00", "по результатам")
        self.low_confidence_card = MetricCard("Низкая уверенность", "0", "ниже 0.60")
        for column, card in enumerate([self.total_card, self.processed_card, self.confidence_card, self.low_confidence_card]):
            metrics.addWidget(card, 0, column)
            metrics.setColumnStretch(column, 1)
        batch_layout.addLayout(metrics)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_results_panel())
        splitter.addWidget(self._build_analysis_right_panel())
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, True)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([820, 320])
        batch_layout.addWidget(splitter, 1)

        self.analysis_mode_stack.addWidget(batch_page)
        self.analysis_mode_stack.addWidget(self._build_quick_analysis_panel())
        page.layout().addWidget(self.analysis_mode_stack, 1)
        return page

    def _build_quick_analysis_panel(self) -> QWidget:
        panel = Panel()
        row = QHBoxLayout()
        row.setSpacing(SPACE_3)

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
        result_box.setMaximumWidth(360)
        result_layout = QVBoxLayout(result_box)
        result_layout.setContentsMargins(SPACE_3, SPACE_3, SPACE_3, SPACE_3)
        result_layout.setSpacing(SPACE_3)

        result_title = QLabel("Результат")
        result_title.setObjectName("formLabel")

        self.quick_result_label = QLabel("Результат появится здесь")
        self.quick_result_label.setObjectName("quickResult")
        self.quick_result_label.setProperty("sentiment", "none")
        self.quick_result_label.setWordWrap(True)
        self.quick_probability_label = QLabel("Вероятности появятся после анализа")
        self.quick_probability_label.setObjectName("mutedLabel")
        self.quick_probability_label.setWordWrap(True)

        button_row = QHBoxLayout()
        button_row.setSpacing(SPACE_2)
        self.quick_analyze_button = QPushButton("Проверить текст")
        self.quick_analyze_button.clicked.connect(self.run_quick_text_analysis)
        clear_button = QPushButton("Очистить")
        clear_button.clicked.connect(self.clear_quick_text_analysis)
        button_row.addWidget(self.quick_analyze_button, 1)
        button_row.addWidget(clear_button)

        result_layout.addWidget(result_title)
        result_layout.addWidget(self.quick_result_label)
        result_layout.addWidget(self.quick_probability_label)
        result_layout.addStretch(1)
        result_layout.addLayout(button_row)
        row.addWidget(input_box, 1)
        row.addWidget(result_box)

        panel.layout.addLayout(row)
        return panel

    def _build_results_panel(self) -> QWidget:
        panel = Panel("Результаты анализа")
        header = QHBoxLayout()
        self.result_count_label = QLabel("Всего: 0")
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Фильтр по тексту, классу или источнику")
        self.filter_edit.setMinimumWidth(220)
        self.filter_edit.textChanged.connect(self.apply_filter)
        header.addWidget(self.result_count_label)
        header.addStretch(1)
        header.addWidget(self.filter_edit)
        panel.layout.addLayout(header)

        self.results_table = QTableWidget(0, 5)
        self._configure_embedded_table(self.results_table)
        self.results_table.setHorizontalHeaderLabels(["#", "Текст", "Класс", "Увер.", "Источник"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.results_table.setColumnWidth(0, 45)
        self.results_table.setColumnWidth(2, 125)
        self.results_table.setColumnWidth(3, 80)
        self.results_table.setColumnWidth(4, 120)
        panel.layout.addWidget(self.results_table, 1)
        return panel

    def _build_analysis_right_panel(self) -> QWidget:
        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_3)
        self.class_chart = ChartWidget("Распределение классов", 155)
        self.confidence_chart = ChartWidget("Распределение уверенности", 155)
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
        splitter = QSplitter(Qt.Orientation.Horizontal)

        settings = Panel("Обучение")
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
        self.train_local_only_check = QCheckBox("Только локальные файлы")
        self.train_local_only_check.setChecked(True)
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
        self.train_truncation_strategy_combo = QComboBox()
        self.train_truncation_strategy_combo.addItems(["longest_first", "only_first", "only_second"])
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
                ("Truncation strategy", self.train_truncation_strategy_combo),
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
        self.train_device_combo.addItems(["Auto", "CPU", "CUDA (0)"])
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
        self.train_button.clicked.connect(self.start_training)
        settings.layout.addWidget(self.train_button)
        settings.setMinimumWidth(660)
        splitter.addWidget(settings)

        progress_panel = Panel("Ход обучения")
        progress_panel.setObjectName("workbenchPanel")
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 1)
        self.training_progress.setValue(0)
        self.training_log = QPlainTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setPlaceholderText("Здесь появятся сообщения обучения.")
        progress_panel.layout.addWidget(self.training_progress)
        progress_panel.layout.addWidget(self.training_log, 1)
        splitter.addWidget(progress_panel)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([680, 560])
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

        auto_button = QPushButton("Автонастроить")
        auto_button.clicked.connect(self.auto_configure_label_mapping)
        apply_button = QPushButton("Проверить")
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

        self.label_mapping_table = QTableWidget(0, 3)
        self._configure_embedded_table(self.label_mapping_table)
        self.label_mapping_table.setHorizontalHeaderLabels(["Метка", "Кол-во", "Действие"])
        self.label_mapping_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.label_mapping_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.label_mapping_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.label_mapping_table.setColumnWidth(1, 64)
        self.label_mapping_table.setColumnWidth(2, 176)
        self.label_mapping_table.verticalHeader().setDefaultSectionSize(40)
        self.label_mapping_table.verticalHeader().setMinimumSectionSize(40)
        layout.addWidget(self.label_mapping_table, 1)

        self.label_mapping_summary = QLabel("Загрузите датасет и выберите колонку метки.")
        self.label_mapping_summary.setObjectName("mappingSummary")
        self.label_mapping_summary.setWordWrap(True)
        self.label_mapping_summary.hide()
        return content

    def _build_models_page(self) -> QWidget:
        page = self._page()
        panel = Panel("Модели и справочные профили")
        note = QLabel(
            "Accuracy, Macro F1 и скорость являются справочными показателями профиля, "
            "а не результатом текущего анализа."
        )
        note.setWordWrap(True)
        note.setObjectName("mutedLabel")
        panel.layout.addWidget(note)
        self.profile_table = QTableWidget(0, 6)
        self._configure_embedded_table(self.profile_table)
        self.profile_table.setHorizontalHeaderLabels(["Модель", "Статус", "Accuracy", "Macro F1", "Уверенность", "Скорость"])
        self.profile_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.profile_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.profile_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.profile_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.profile_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.profile_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.profile_table.setColumnWidth(1, 112)
        self.profile_table.setColumnWidth(2, 88)
        self.profile_table.setColumnWidth(3, 88)
        self.profile_table.setColumnWidth(4, 108)
        self.profile_table.setColumnWidth(5, 96)
        self.profile_table.cellDoubleClicked.connect(self._select_model_from_profile)
        panel.layout.addWidget(self.profile_table, 1)
        page.layout().addWidget(panel, 1)
        return page

    def _build_monitoring_page(self) -> QWidget:
        page = self._page()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.drift_chart = ChartWidget("Динамика уверенности и распределений", 320)
        splitter.addWidget(self.drift_chart)
        panel = Panel("Сводка мониторинга")
        self.monitoring_summary_label = QLabel("Недостаточно данных для мониторинга.")
        self.monitoring_summary_label.setWordWrap(True)
        self.monitoring_summary_label.setObjectName("mutedLabel")
        self.monitoring_table = QTableWidget(0, 2)
        self._configure_embedded_table(self.monitoring_table)
        self.monitoring_table.setHorizontalHeaderLabels(["Показатель", "Значение"])
        self.monitoring_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.monitoring_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        panel.layout.addWidget(self.monitoring_summary_label)
        panel.layout.addWidget(self.monitoring_table, 1)
        splitter.addWidget(panel)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, True)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([760, 320])
        page.layout().addWidget(splitter, 1)
        return page

    def _build_reports_page(self) -> QWidget:
        page = self._page()
        panel = Panel("Экспорт отчета")
        text = QLabel("HTML-отчет включает сведения о выбранной модели, сводку результатов и первые 500 строк анализа.")
        text.setWordWrap(True)
        text.setObjectName("mutedLabel")
        export_button = QPushButton("Экспортировать HTML-отчет")
        export_button.clicked.connect(self.export_report)
        panel.layout.addWidget(text)
        panel.layout.addWidget(export_button)
        panel.layout.addStretch(1)
        page.layout().addWidget(panel)
        page.layout().addStretch(1)
        return page

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
        table.horizontalHeader().setHighlightSections(False)
        table.verticalHeader().setVisible(False)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _set_status(self, text: str, state: str = "ready") -> None:
        self.status_label.setText(text)
        self.status_label.setProperty("state", state)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _set_initial_state(self) -> None:
        self.analyze_button.setEnabled(False)
        self.train_button.setEnabled(False)
        self._switch_page(0)
        self.populate_profile_table()
        self._refresh_empty_charts()
        self._log("Готово. Загрузите датасет для начала работы.")
        self.refresh_label_mapping_table()

    def _switch_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        if 0 <= index < len(self.nav_buttons):
            self.nav_buttons[index].setChecked(True)

    def _switch_analysis_mode(self, index: int) -> None:
        self.analysis_mode_stack.setCurrentIndex(index)

    def load_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить набор текстовых данных",
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
        self.dataset_name_label.setText(self.dataset_path.name)
        self.dataset_name_label.setToolTip(str(self.dataset_path))
        self.dataset_rows_label.setText(f"строк: {format_int(len(self.data_frame))}")
        self._populate_column_combo()
        self._populate_training_columns()
        self.results = []
        self.preview_offset = 0
        self.populate_preview_table()
        self.populate_results_table([])
        self.refresh_analysis()
        self.refresh_label_mapping_table()
        self.analyze_button.setEnabled(True)
        self.train_button.setEnabled(True)
        self.data_summary_label.setText(
            f"{self.dataset_path.name}: {format_int(len(self.data_frame))} строк, {len(self.data_frame.columns)} колонок."
        )
        self.dataset_rows_card.set_values(format_int(len(self.data_frame)), "в наборе данных")
        self.dataset_columns_card.set_values(format_int(len(self.data_frame.columns)), "доступно для выбора")
        self._set_status("Данные загружены", "ready")
        self.statusBar().showMessage("Данные загружены. Выберите колонку и запустите анализ или настройте метки для обучения.")
        self._log(f"Данные загружены: {self.dataset_path.name} ({len(self.data_frame)} строк).")
        self._switch_page(0)

    def run_analysis(self) -> None:
        if self.data_frame.empty:
            QMessageBox.information(self, "Нет данных", "Сначала загрузите CSV, TXT или Excel файл.")
            return
        text_column = self.text_column_combo.currentText()
        if text_column not in self.data_frame.columns:
            QMessageBox.warning(self, "Колонка не найдена", "Выберите колонку с текстом.")
            return

        source_column = self._guess_source_column()
        options = self._analysis_options()
        self._set_status("Загрузка модели", "running")
        self.statusBar().showMessage("Загрузка transformer-модели...")
        QApplication.processEvents()

        try:
            analyzer = SentimentAnalyzer(self.model_combo.currentText(), options, self.device_combo.currentText())
            self._log(f"Transformer-модель загружена: {self.model_combo.currentText()}.")
            self._set_status("Анализ", "running")
            self.statusBar().showMessage("Выполняется анализ...")
            QApplication.processEvents()
            self.results = analyzer.analyze(
                self.data_frame[text_column].fillna("").astype(str).tolist(),
                self.data_frame[source_column].fillna("").astype(str).tolist() if source_column else None,
            )
        except TransformerLoadError as exc:
            self._set_status("Ошибка модели", "error")
            self.statusBar().showMessage("Ошибка transformer-модели.")
            self._log(f"Ошибка transformer-модели: {exc}")
            QMessageBox.critical(self, "Ошибка transformer-модели", str(exc))
            return

        self._set_status("Готово", "ready")
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
            analyzer = SentimentAnalyzer(self.model_combo.currentText(), self._analysis_options(), self.device_combo.currentText())
            result = analyzer.analyze([text])[0]
        except TransformerLoadError as exc:
            self._set_status("Ошибка модели", "error")
            self.statusBar().showMessage("Ошибка transformer-модели.")
            self._log(f"Ошибка быстрого анализа: {exc}")
            QMessageBox.critical(self, "Ошибка transformer-модели", str(exc))
            return

        self.quick_result_label.setText(f"{result.sentiment} · уверенность {result.confidence:.2f}")
        self._set_quick_result_sentiment(result.sentiment)
        self.quick_probability_label.setText(format_probabilities(result.probabilities))
        self._set_status("Готово", "ready")
        self.statusBar().showMessage("Быстрый анализ текста завершен.")
        self._log(f"Быстрый анализ: {result.sentiment}, уверенность {result.confidence:.2f}.")

    def clear_quick_text_analysis(self) -> None:
        self.quick_text_edit.clear()
        self.quick_result_label.setText("Результат появится здесь")
        self._set_quick_result_sentiment(None)
        self.quick_probability_label.setText("Вероятности появятся после анализа")

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

    def export_report(self) -> None:
        if not self.results:
            QMessageBox.information(self, "Нет результатов", "Сначала выполните анализ.")
            return
        default_name = f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт отчета", default_name, "HTML (*.html)")
        if not path:
            return
        report_path = export_html_report(path, self.results, self.model_combo.currentText())
        self._log(f"HTML-отчет сохранен: {report_path}.")
        self.statusBar().showMessage(f"Отчет сохранен: {report_path}")

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
            stratify_split=self.train_stratify_check.isChecked(),
            split_seed=self.train_split_seed_spin.value(),
            seed=self.train_seed_spin.value(),
            max_length=self.train_max_length_spin.value(),
            padding=self.train_padding_combo.currentText(),
            truncation=self.train_truncation_check.isChecked(),
            truncation_strategy=self.train_truncation_strategy_combo.currentText(),
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
        self.training_log.appendPlainText("Запуск обучения...")
        self.training_log.appendPlainText(
            f"После настройки меток: используется {prepared.used_rows} строк, исключено {prepared.excluded_rows}."
        )
        self.training_log.appendPlainText(
            "Классы: " + ", ".join(f"{label}: {count}" for label, count in prepared.class_counts.items())
        )
        self.training_progress.setRange(0, 0)
        self.train_button.setEnabled(False)
        self._set_status("Обучение", "running")
        self.statusBar().showMessage("Выполняется fine-tuning transformer-модели...")

        self.training_thread = TrainingThread(prepared.texts, prepared.labels, config)
        self.training_thread.message.connect(self.training_log.appendPlainText)
        self.training_thread.completed.connect(self.on_training_completed)
        self.training_thread.failed.connect(self.on_training_failed)
        self.training_thread.start()

    def on_training_completed(self, result: TrainResult) -> None:
        self.training_progress.setRange(0, 1)
        self.training_progress.setValue(1)
        self.train_button.setEnabled(True)
        self._set_status("Готово", "ready")
        self.training_log.appendPlainText(
            f"Готово: accuracy={result.accuracy:.3f}, macro_f1={result.macro_f1:.3f}, "
            f"train={result.train_size}, validation={result.validation_size}"
        )
        output_path = str(result.output_dir)
        if self.model_combo.findText(output_path) == -1:
            self.model_combo.addItem(output_path)
        self.model_combo.setCurrentText(output_path)
        self.populate_profile_table()
        self.statusBar().showMessage(f"Модель обучена и сохранена: {output_path}")
        self._log(f"Обучение завершено. Модель сохранена: {output_path}.")

    def on_training_failed(self, message: str) -> None:
        self.training_progress.setRange(0, 1)
        self.training_progress.setValue(0)
        self.train_button.setEnabled(True)
        self._set_status("Ошибка обучения", "error")
        self.training_log.appendPlainText(message)
        self.statusBar().showMessage("Ошибка обучения.")
        QMessageBox.critical(self, "Ошибка обучения", message)

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

        self.label_action_combos.clear()
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
        for value in self.data_frame[label_column].tolist():
            for token in parse_label_tokens(value, parse_as_list=parse_as_list):
                self.label_count_by_token[token] += 1

        rows = sorted(self.label_count_by_token.items(), key=lambda item: (-item[1], label_key(item[0])))
        header = "Sentiment" if self.target_scheme_combo.currentText() == SENTIMENT_SCHEME else "Действие"
        self.label_mapping_table.setHorizontalHeaderLabels(["Метка", "Кол-во", header])
        self.label_mapping_table.setRowCount(len(rows))
        for row, (raw_label, count) in enumerate(rows):
            self.label_mapping_table.setItem(row, 0, QTableWidgetItem(raw_label))
            self.label_mapping_table.setItem(row, 1, QTableWidgetItem(format_int(count)))

            action = self._default_label_action(raw_label)
            combo = make_combo(self._label_action_values(), action)
            combo.setObjectName("tableCombo")
            combo.currentTextChanged.connect(self.update_label_mapping_summary)
            self.label_action_combos[raw_label] = combo
            self.label_mapping_table.setCellWidget(row, 2, combo)
            self.label_mapping_table.setRowHeight(row, 40)

        self.update_label_mapping_summary()

    def auto_configure_label_mapping(self) -> None:
        if not hasattr(self, "label_mapping_table"):
            return
        for raw_label, combo in self.label_action_combos.items():
            action = self._default_label_action(raw_label)
            combo.setCurrentText(action)
        self.update_label_mapping_summary()

    def _current_label_mapping(self) -> dict[str, str]:
        return {raw_label: combo.currentText() for raw_label, combo in self.label_action_combos.items()}

    def _map_tokens_to_target(self, tokens: list[str], mapping: dict[str, str]) -> str | None:
        if not tokens:
            return None

        strategy = self.multilabel_strategy_combo.currentText()
        parse_as_list = self.label_format_combo.currentText() == "Список меток в строке"

        if parse_as_list and len(tokens) > 1 and strategy == "Исключать строки со списком меток":
            return None

        if parse_as_list and len(tokens) > 1 and strategy == "Первая метка в списке":
            tokens = tokens[:1]

        for token in tokens:
            action = mapping.get(token, auto_target_for_label(token))
            if action == EXCLUDE_LABEL:
                continue
            if action == KEEP_ORIGINAL:
                return token
            return action
        return None

    def _prepare_training_data(self) -> LabelPrepareResult:
        text_column = self.train_text_column_combo.currentText()
        label_column = self.train_label_column_combo.currentText()
        parse_as_list = self.label_format_combo.currentText() == "Список меток в строке"
        mapping = self._current_label_mapping()

        texts: list[str] = []
        labels: list[str] = []
        excluded = 0

        for _, row in self.data_frame.iterrows():
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
        self.label_mapping_summary.setText(
            f"После настройки меток будет использовано {format_int(prepared.used_rows)} строк, "
            f"исключено {format_int(prepared.excluded_rows)} строк. Классы: {counts}.{list_note}"
        )

    def refresh_analysis(self) -> None:
        summary = summarize_results(self.results)
        total = len(self.data_frame)
        processed = int(summary["processed"])
        low_confidence = sum(result.confidence < 0.60 for result in self.results)
        self.total_card.set_values(format_int(total), "в наборе данных")
        self.processed_card.set_values(format_int(processed), f"строк ({processed / max(total, 1):.1%})")
        self.confidence_card.set_values(f"{float(summary['avg_confidence']):.2f}", "по результатам")
        self.low_confidence_card.set_values(format_int(low_confidence), "ниже 0.60")
        self.populate_results_table(self.results)
        self.class_chart.draw_class_distribution(class_distribution(self.results))
        self.confidence_chart.draw_confidence_histogram(self.results)
        self.refresh_monitoring()

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
            ("Низкая уверенность (<0.60)", format_int(sum(result.confidence < 0.60 for result in self.results))),
        ]
        rows.extend((f"Класс: {label}", format_int(count)) for label, count in counts.most_common(8))
        self.monitoring_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.monitoring_table.setItem(row, 0, QTableWidgetItem(name))
            self.monitoring_table.setItem(row, 1, QTableWidgetItem(value))

    def populate_preview_table(self) -> None:
        total = len(self.data_frame)
        page_size = self.preview_page_size_spin.value() if hasattr(self, "preview_page_size_spin") else 200
        if total == 0:
            self.preview_offset = 0
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            self.dataset_preview_card.set_values("0", "строк в предпросмотре")
            if hasattr(self, "preview_range_label"):
                self.preview_range_label.setText("Строки 0-0 из 0")
                self.preview_jump_spin.setRange(1, 1)
                for button in (self.preview_first_button, self.preview_prev_button, self.preview_next_button):
                    button.setEnabled(False)
            return

        self.preview_offset = max(0, min(self.preview_offset, max(total - 1, 0)))
        frame = self.data_frame.iloc[self.preview_offset : self.preview_offset + page_size]
        self.preview_table.setRowCount(len(frame))
        self.preview_table.setColumnCount(len(frame.columns))
        self.preview_table.setHorizontalHeaderLabels([str(column) for column in frame.columns])
        header = self.preview_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        if len(frame.columns) > 0:
            preferred = self.text_column_combo.currentText()
            preferred_index = list(frame.columns).index(preferred) if preferred in frame.columns else 0
            header.setSectionResizeMode(preferred_index, QHeaderView.ResizeMode.Stretch)
        for row_index, (_, row) in enumerate(frame.iterrows()):
            for column_index, value in enumerate(row):
                self.preview_table.setItem(row_index, column_index, QTableWidgetItem(str(value)[:300]))
        start = self.preview_offset + 1
        end = self.preview_offset + len(frame)
        self.dataset_preview_card.set_values(format_int(len(frame)), f"строки {format_int(start)}-{format_int(end)}")
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
        if self.data_frame.empty:
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
        if self.data_frame.empty:
            return
        page_size = self.preview_page_size_spin.value()
        self.preview_offset = min(max(len(self.data_frame) - 1, 0), self.preview_offset + page_size)
        self.populate_preview_table()

    def populate_results_table(self, results: list[AnalysisResult]) -> None:
        visible_results = results[:1000]
        self.result_count_label.setText(f"Всего: {format_int(len(results))}")
        score_labels = result_score_labels(visible_results)
        headers = ["#", "Текст", "Класс", "Увер.", *score_labels, "Источник"]
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        if len(headers) > 1:
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.results_table.setColumnWidth(0, 45)
        self.results_table.setColumnWidth(2, 128)
        self.results_table.setColumnWidth(3, 80)
        for column in range(4, 4 + len(score_labels)):
            self.results_table.setColumnWidth(column, 96)
        self.results_table.setColumnWidth(len(headers) - 1, 120)
        self.results_table.setRowCount(len(visible_results))
        for row, result in enumerate(visible_results):
            values = [
                str(row + 1),
                result.text[:260],
                result.sentiment,
                f"{result.confidence:.2f}",
                *[f"{result.probabilities.get(label, 0.0):.2f}" for label in score_labels],
                result.source,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 2:
                    item.setForeground(QColor(SENTIMENT_COLORS.get(result.sentiment, "#111827")))
                self.results_table.setItem(row, column, item)

    def populate_profile_table(self) -> None:
        if not hasattr(self, "profile_table"):
            return
        rows = comparison_rows()
        self.profile_table.setRowCount(len(rows))
        for row, profile in enumerate(rows):
            name = str(profile["name"])
            values = [
                name,
                self._model_status(name),
                f"{float(profile['accuracy']):.2f}",
                f"{float(profile['macro_f1']):.2f}",
                f"{float(profile['confidence']):.2f}",
                f"{profile['speed']} стр/с",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if name == self.model_combo.currentText():
                    item.setBackground(QColor("#d7e5fb"))
                self.profile_table.setItem(row, column, item)

    def apply_filter(self, text: str) -> None:
        if not text:
            self.populate_results_table(self.results)
            return
        query = text.casefold()
        filtered = [
            result
            for result in self.results
            if query in result.text.casefold() or query in result.sentiment.casefold() or query in result.source.casefold()
        ]
        self.populate_results_table(filtered)

    def _select_model_from_profile(self, row: int, column: int) -> None:
        if self.profile_table.item(row, 0) is not None:
            self.model_combo.setCurrentText(self.profile_table.item(row, 0).text())
            self.statusBar().showMessage("Модель выбрана из профиля.")

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

    def _populate_column_combo(self) -> None:
        self.text_column_combo.clear()
        columns = [str(column) for column in self.data_frame.columns]
        preferred = next(
            (column for column in columns if column.lower() in {"text", "текст", "review", "content", "message"}),
            columns[0] if columns else "text",
        )
        self.text_column_combo.addItems(columns)
        self.text_column_combo.setCurrentText(preferred)
        self._log(f"Колонка текста: {preferred}.")

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

    def _guess_source_column(self) -> str | None:
        for column in self.data_frame.columns:
            if str(column).lower() in {"source", "источник", "platform", "site"}:
                return str(column)
        return None

    @staticmethod
    def _model_status(name: str) -> str:
        if Path(name).exists():
            return "локальная"
        if name.startswith("models/"):
            return "не найдена"
        return "профиль HF"

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
            #contextBar {
                background: #f8fafc;
                border-bottom: 1px solid rgba(129, 145, 166, 0.35);
                border-radius: 0;
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
                font-weight: 400;
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
            #mappingSummary {
                background: #f8fafc;
                border: 1px solid rgba(129, 145, 166, 0.32);
                border-radius: 4px;
                color: #344054;
                padding: 8px;
                font-weight: 400;
            }
            #quickResultPanel {
                background: #fbfcfe;
                border-left: 1px solid rgba(129, 145, 166, 0.38);
                border-radius: 0;
            }
            QPlainTextEdit#quickTextInput {
                min-height: 240px;
            }
            #quickResult {
                color: #111827;
                font-size: 13px;
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
                font-weight: 400;
            }
            QPushButton#primaryButton:hover { background: #1f65b8; }
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
                min-height: 28px;
                max-height: 28px;
                margin: 4px;
                padding: 4px 8px;
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
                padding: 8px;
                font-weight: 400;
                font-size: 11px;
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
                border-radius: 4px;
                padding: 8px;
                font-weight: 400;
            }
            QLabel#statusChip[state="ready"] {
                background: #eef9ee;
                border: 1px solid rgba(34, 126, 62, 0.35);
                color: #166534;
            }
            QLabel#statusChip[state="running"] {
                background: #eef5ff;
                border: 1px solid rgba(37, 111, 199, 0.35);
                color: #1f5fa9;
            }
            QLabel#statusChip[state="error"] {
                background: #fef2f2;
                border: 1px solid rgba(220, 38, 38, 0.34);
                color: #991b1b;
            }
            """
        )
