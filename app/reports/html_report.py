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
    top_cards = "".join(
        f"<div class=\"card\">{escape(label)}<div class=\"value\">{count}</div></div>"
        for label, count in counts.most_common(3)
    )
    rows = "\n".join(
        f"<tr><td>{index}</td><td>{escape(result.text[:220])}</td><td>{result.sentiment}</td>"
        f"<td>{result.confidence:.2f}</td><td>{escape(result.source)}</td></tr>"
        for index, result in enumerate(results[:500], start=1)
    )
    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Отчет анализа тональности</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #64748b; margin-bottom: 24px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
    .card {{ border: 1px solid #dbe3ef; border-radius: 8px; padding: 16px; }}
    .value {{ font-size: 26px; font-weight: 700; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <h1>Отчет анализа тональности</h1>
  <div class="meta">Сформировано: {datetime.now().strftime("%d.%m.%Y %H:%M:%S")} | Модель: {escape(model_name)}</div>
  <div class="cards">
    <div class="card">Всего текстов<div class="value">{len(results)}</div></div>
    <div class="card">Средняя уверенность<div class="value">{avg_confidence:.2f}</div></div>
    {top_cards}
  </div>
  <h2>Первые 500 результатов</h2>
  <table>
    <thead><tr><th>#</th><th>Текст</th><th>Тональность</th><th>Уверенность</th><th>Источник</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return report_path
