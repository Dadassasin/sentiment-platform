# Design System: Sentiment Platform

## Direction
- Feel: Dense, quiet desktop laboratory for sentiment analysis and transformer training.
- Primary user: Analyst or researcher reviewing datasets, model outputs, training runs, and drift signals.
- Primary task: Load a dataset, run sentiment analysis, inspect low-confidence rows, and compare model behavior.
- Signature: Workbench layout with a persistent analysis pipeline sidebar, result tables as the main surface, and compact monitoring charts as secondary evidence.

## Tokens
- Spacing base: 4px
- Spacing scale: 4px, 8px, 12px, 16px
- Radius scale: 4px, 6px
- Depth: Borders-only

## Colors
- Canvas: #eef2f7
- Surface: #ffffff
- Surface inset: #fbfcfe
- Surface toolbar: #f8fafc
- Surface muted: #f3f6fa
- Chart accent: #7aa5dc
- Text primary: #111827
- Text body: #172033
- Text strong: #1d2a3d
- Text control: #1f2937
- Text nav: #344054
- Text secondary: #5b677a
- Text section: #7b8798
- Text header: #526070
- Text muted: #6b7280
- Text disabled: #9ca3af
- Border: rgba(129, 145, 166, 0.38)
- Border soft: rgba(129, 145, 166, 0.32)
- Border control: rgba(107, 120, 140, 0.58)
- Accent: #256fc7
- Accent sentiment positive: #2563eb
- Accent border: #1f5fa9
- Accent focus: #6e9bd6
- Accent hover soft: #eef5ff
- Accent hover: #1f65b8
- Accent pressed: #1a559b
- Accent pressed soft: #e1ecfb
- Accent soft: #dbeafe
- Accent selected: #d7e5fb
- Success: #166534
- Warning: #d97706
- Danger: #dc2626
- Danger text: #991b1b

## Typography
- Heading: 13px, regular weight
- Body: 12px, regular weight
- Label: 11px, regular weight
- Data: 22px, regular weight for metric values

## Patterns
### Panel
- Padding: 12px top/side, 8px title gap
- Radius: 6px
- Border: 1px low-opacity slate
- Depth: Borders-only

### Metric Card
- Minimum height: 76px
- Padding: 12px horizontal, 8px vertical
- Radius: 6px
- Value: 22px, 700 weight

### Button
- Height: 28px minimum
- Padding: 4px 12px
- Radius: 4px
- Font: 12px, 600 weight

### Input
- Height: 28px minimum
- Padding: 4px 8px
- Radius: 4px
- Focus: accent border only

### Dataset Chip
- Role: Read-only active dataset indicator in the toolbar
- Height: 28px minimum
- Padding: 4px 8px
- Radius: 4px
- Surface: canvas tint with soft border

### Embedded Table
- Border: none inside panels
- Background: #fbfcfe
- Header: #f3f6fa, 8px padding, 11px 700 weight
- Selection: #dbeafe with primary text

### Status Chip
- Padding: 8px
- Radius: 4px
- States: ready green, running blue, error red

### Quick Text Analysis
- Placement: Separate "Один текст" analysis mode, not above batch results
- Input: QPlainTextEdit, 76px minimum height, 116px maximum height
- Result: 13px, 700 weight, sentiment color via `sentiment` dynamic property
- Actions: Secondary "Проверить текст" and secondary "Очистить"

### Analysis Mode Switch
- Placement: Top of analysis page
- Modes: "Пакетный анализ", "Один текст"
- Height: 28px minimum
- Checked state: accent soft background, accent text
- Focus: accent border

### Focus States
- Buttons: accent border and soft accent background
- Primary buttons: dark focus border on accent surface
- Text inputs: accent border
- Checkboxes: accent text

### Training Workbench
- Placement: Single `Обучение` app page with internal tabs.
- Tabs: short labels only, e.g. `Проект`, `Данные`, `Метки`, `Выборка`, `Модель`, `Токены`, `Тренинг`, `Оптим.`, `Доп.`.
- Training settings panel minimum width: 660px so tabs fit without Qt scroll buttons.
- Do not use `QTabWidget` for the training workbench; PyQt tab sizing does not reliably fill width during splitter drag.
- Training tabs use a custom button row plus `QStackedWidget`; every tab button has equal layout stretch and expands with the panel.
- Purpose: Keep all transformer training settings in one page without one long form.
- Layout: Left settings panel with scrollable tab forms, right progress/log panel.
- Field density: 28px controls, 12px row gaps, label column plus flexible control column.
- Training device is configured inside the training page; toolbar device is for inference.
- Only expose settings wired into `TrainConfig`; unsupported active learning remains absent.
- Every training field has visible helper text plus a tooltip; no unexplained ML parameters.

### Context Bar
- Role: Dataset context only, not a duplicate of analysis or training tabs.
- Contains: dataset loading, active file, row count, and inference device.
- Does not contain a status chip, analysis text column selector, or analysis model selector.
- Analysis text column and analysis model live inside the `Анализ` page.
- Training model, training dataset mapping, and training device stay inside the `Обучение` page.

### Label Mapping
- Placement: `Обучение` page, `Метки` tab.
- Supports single-label cells and list-like cells such as `(3, 5, 7)`, `[positive, skip]`, or comma-separated values.
- Default scheme keeps selected source labels as their own classes, which supports emotion/topic datasets without forcing sentiment classes.
- Sentiment mapping is an explicit alternate scheme; only then can source labels be mapped to positive, neutral, or negative.
- Skip-like labels (`skip`, `other`, `unknown`, empty values) default to excluded.
- Use compact labels and tooltips in this tab; do not show long helper paragraphs under controls.
- Table headers stay short: `Метка`, `Кол-во`, `Действие` or `Sentiment`.
- Do not place explanatory labels under the mapping table; keep vertical space for the table itself.

### Splitters
- Sidebar is fixed at 188px and has no draggable handle.
- Internal two-panel work areas may use a horizontal splitter.
- Splitter handle: 4px transparent hit area with no visible line.
- Training workbench panels use 6px radius; splitter has no visible line.

### Model Profiles Table
- Show only real inference profiles, not placeholder local paths for models that are not present.
- Headers are compact and readable: `Модель`, `Статус`, `Accuracy`, `Macro F1`, `Уверенность`, `Скорость`.
- Metric columns use fixed widths; model column stretches.

### Analysis Page
- Right column is reserved for charts and evidence, not an event log.
- Runtime events are kept in memory and surfaced through the status bar.
- Analysis settings use compact horizontal rows, not a stretched grid with large empty gaps.
- Model selector may stretch; preprocessing checkboxes stay grouped tightly before the primary action.
- Result table probability columns are dynamic and come from actual model labels, not fixed `Positive/Neutral/Negative`.
- Quick text analysis shows whatever probability labels the loaded model returns.

### Dataset Preview
- Preview is paginated, not a static first-200-rows table.
- Controls include current range, page size, jump-to-row, first page, previous page, and next page.
- Preview metric shows the current page size and row range.

### Quick Text Analysis
- Input area is the primary surface and should be tall enough for paragraph text.
- Result is a separate right panel with a left border, not loose text floating beside the input.
- Result panel includes title, predicted class, probabilities, and actions.
