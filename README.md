# Sentiment Platform

Исследовательская десктопная платформа для обучения, применения и сравнения нейросетевых моделей анализа тональности русскоязычных текстов на Python и PyQt6.

## Возможности

- загрузка CSV, TXT и Excel-файлов;
- пакетный анализ тональности текстов;
- быстрая проверка одного текста;
- настройка предобработки русского текста;
- обучение transformer-классификатора на пользовательских метках;
- сравнение нескольких локальных моделей;
- простой активный отбор неуверенных примеров для ручной разметки;
- мониторинг результата анализа: уверенность, перекос классов, предупреждения и сравнение с предыдущим запуском;
- экспорт HTML-отчета.

## Требования

Рекомендуется Python `3.10-3.13`.

Python `3.14` не рекомендуется, потому что для него могут отсутствовать PyTorch CUDA wheels.

## Установка

Создать виртуальное окружение:

```powershell
python -m venv venv
```

Активировать окружение в PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Установить зависимости:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Скачать NLTK stopwords:

```powershell
python -m nltk.downloader stopwords
```

## Запуск

```powershell
python -m app.main
```