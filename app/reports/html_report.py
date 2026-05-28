"""HTML report generation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path

from app.models.sentiment import AnalysisResult


def export_html_report(path: str | Path, results: list[AnalysisResult], model_name: str) -> Path:
    report_path = Path(path)
    counts = Counter(result.sentiment for result in results)
    avg_confidence = sum(result.confidence for result in results) / len(results) if results else 0
    low_confidence = [result for result in results if result.confidence < 0.60]
    score_labels = probability_labels(results)

    top_cards = "".join(
        f"<div class=\"card\"><span>{escape(label)}</span><div class=\"value\">{format_int(count)}</div></div>"
        for label, count in counts.most_common(4)
    )
    rows = "\n".join(
        f"<tr><td>{index}</td><td>{escape(result.text[:260])}</td><td>{escape(result.sentiment)}</td>"
        f"<td>{result.confidence:.2f}</td>{probability_cells(result, score_labels)}</tr>"
        for index, result in enumerate(results[:500], start=1)
    )
    low_rows = "\n".join(
        f"<tr><td>{escape(result.text[:220])}</td><td>{escape(result.sentiment)}</td><td>{result.confidence:.2f}</td></tr>"
        for result in low_confidence[:30]
    )

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Отчёт анализа тональности</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; background: #ffffff; }}
    h1 {{ margin: 0 0 4px; font-size: 28px; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; }}
    .meta {{ color: #64748b; margin-bottom: 24px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
    .card {{ border: 1px solid #dbe3ef; border-radius: 8px; padding: 16px; background: #fbfcfe; }}
    .card span {{ color: #526070; }}
    .value {{ font-size: 26px; margin-top: 8px; color: #111827; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .panel {{ border: 1px solid #dbe3ef; border-radius: 8px; padding: 16px; background: #ffffff; }}
    .bar-row {{ display: grid; grid-template-columns: 150px 1fr 72px; gap: 10px; align-items: center; margin: 8px 0; }}
    .track {{ height: 12px; background: #edf2f7; border-radius: 999px; overflow: hidden; }}
    .bar {{ height: 12px; background: #7aa5dc; border-radius: 999px; }}
    .hist {{ display: grid; grid-template-columns: repeat(10, 1fr); gap: 6px; align-items: end; height: 160px; }}
    .hist-col {{ background: #7aa5dc; min-height: 2px; border-radius: 4px 4px 0 0; }}
    .hist-labels {{ display: grid; grid-template-columns: repeat(10, 1fr); gap: 6px; color: #64748b; font-size: 11px; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f8fafc; color: #526070; }}
    .muted {{ color: #64748b; }}
  </style>
</head>
<body>
  <h1>Отчёт анализа тональности</h1>
  <div class="meta">Сформировано: {datetime.now().strftime("%d.%m.%Y %H:%M:%S")} | Модель: {escape(model_name)}</div>

  <div class="cards">
    <div class="card"><span>Всего текстов</span><div class="value">{format_int(len(results))}</div></div>
    <div class="card"><span>Средняя уверенность</span><div class="value">{avg_confidence:.2f}</div></div>
    <div class="card"><span>Низкая уверенность</span><div class="value">{format_int(len(low_confidence))}</div></div>
    {top_cards}
  </div>

  <div class="grid">
    <div class="panel">
      <h2>Распределение классов</h2>
      {class_distribution_chart(counts)}
    </div>
    <div class="panel">
      <h2>Распределение уверенности</h2>
      {confidence_histogram(results)}
    </div>
  </div>

  <h2>Примеры с низкой уверенностью</h2>
  {low_confidence_table(low_rows)}

  <h2>Первые 500 результатов</h2>
  <table>
    <thead><tr><th>#</th><th>Текст</th><th>Класс</th><th>Уверенность</th>{probability_headers(score_labels)}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    return report_path


def format_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def class_distribution_chart(counts: Counter[str]) -> str:
    total = sum(counts.values())
    if total == 0:
        return "<p class=\"muted\">Нет данных</p>"

    rows = []
    for label, count in counts.most_common():
        share = count / total
        rows.append(
            "<div class=\"bar-row\">"
            f"<div>{escape(label)}</div>"
            f"<div class=\"track\"><div class=\"bar\" style=\"width:{share:.1%}\"></div></div>"
            f"<div>{format_int(count)} ({share:.0%})</div>"
            "</div>"
        )
    return "".join(rows)


def confidence_histogram(results: list[AnalysisResult]) -> str:
    if not results:
        return "<p class=\"muted\">Нет данных</p>"

    bins = [0] * 10
    for result in results:
        index = min(9, max(0, int(result.confidence * 10)))
        bins[index] += 1
    max_count = max(bins) or 1
    columns = "".join(
        f"<div class=\"hist-col\" title=\"{format_int(count)}\" style=\"height:{max(count / max_count * 100, 2):.0f}%\"></div>"
        for count in bins
    )
    labels = "".join(f"<div>{i / 10:.1f}</div>" for i in range(10))
    return f"<div class=\"hist\">{columns}</div><div class=\"hist-labels\">{labels}</div>"


def low_confidence_table(rows: str) -> str:
    if not rows:
        return "<p class=\"muted\">Нет строк с уверенностью ниже 0.60.</p>"
    return (
        "<table>"
        "<thead><tr><th>Текст</th><th>Класс</th><th>Уверенность</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )


def probability_labels(results: list[AnalysisResult]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for result in results:
        for label in result.probabilities:
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return labels[:8]


def probability_headers(labels: list[str]) -> str:
    return "".join(f"<th>{escape(label)}</th>" for label in labels)


def probability_cells(result: AnalysisResult, labels: list[str]) -> str:
    return "".join(f"<td>{result.probabilities.get(label, 0.0):.2f}</td>" for label in labels)
