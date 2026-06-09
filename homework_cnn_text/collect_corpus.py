"""
Сбор текстового корпуса новостей с Lenta.ru для задачи классификации по рубрикам.

Источник данных: RSS-ленты рубрик https://lenta.ru/rss/news/<rubric>
Разметка: автоматическая — класс соответствует рубрике RSS-ленты (метка с сайта).
"""

from __future__ import annotations

import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Рубрики Lenta.ru и числовые метки классов
RUBRICS: dict[str, int] = {
    "sport": 0,       # Спорт
    "economics": 1,   # Экономика
    "science": 2,     # Наука
    "culture": 3,     # Культура
}

CLASS_NAMES = {
    0: "sport",
    1: "economics",
    2: "science",
    3: "culture",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_CSV = DATA_DIR / "corpus.csv"
MIN_SAMPLES = 500
REQUEST_TIMEOUT = 20
MAX_WORKERS = 6


def fetch_rss_items(rubric: str) -> list[dict[str, str]]:
    """Загружает заголовки и URL статей из RSS-ленты рубрики."""
    url = f"https://lenta.ru/rss/news/{rubric}"
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "xml")
    items: list[dict[str, str]] = []
    for item in soup.find_all("item"):
        link_tag = item.find("link")
        title_tag = item.find("title")
        if not link_tag or not title_tag:
            continue
        article_url = link_tag.get_text(strip=True)
        title = title_tag.get_text(strip=True)
        if article_url and title:
            items.append({"url": article_url, "title": title})
    return items


def fetch_article_text(url: str, fallback_title: str) -> str:
    """Загружает заголовок и первые абзацы статьи для формирования текста примера."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "lxml")

        title_tag = soup.select_one("h1")
        title = title_tag.get_text(strip=True) if title_tag else fallback_title

        paragraphs = [
            p.get_text(strip=True)
            for p in soup.select("div.topic-body__content p")
            if p.get_text(strip=True)
        ]
        body = " ".join(paragraphs[:3])
        text = f"{title}. {body}" if body else title
        return re.sub(r"\s+", " ", text).strip()
    except requests.RequestException:
        return fallback_title


def collect_sample(args: tuple[str, int, dict[str, str]]) -> dict[str, str | int]:
    """Обрабатывает одну статью: скачивает текст и возвращает размеченный пример."""
    rubric, label, item = args
    text = fetch_article_text(item["url"], item["title"])
    time.sleep(0.15)  # бережная нагрузка на сервер
    return {
        "text": text,
        "label": label,
        "class_name": rubric,
        "url": item["url"],
    }


def collect_corpus() -> list[dict[str, str | int]]:
    """Собирает корпус из всех рубрик и возвращает список примеров."""
    raw_items: list[tuple[str, int, dict[str, str]]] = []
    seen_urls: set[str] = set()

    for rubric, label in RUBRICS.items():
        print(f"Загрузка RSS: {rubric}...")
        for item in fetch_rss_items(rubric):
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                raw_items.append((rubric, label, item))

    print(f"Найдено уникальных статей: {len(raw_items)}")
    print("Загрузка текстов статей (параллельно)...")

    samples: list[dict[str, str | int]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(collect_sample, args) for args in raw_items]
        for i, future in enumerate(as_completed(futures), start=1):
            samples.append(future.result())
            if i % 50 == 0 or i == len(futures):
                print(f"  обработано {i}/{len(futures)}")

    return samples


def save_corpus(samples: list[dict[str, str | int]], path: Path) -> None:
    """Сохраняет корпус в CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "class_name", "url"])
        writer.writeheader()
        writer.writerows(samples)


def print_stats(samples: list[dict[str, str | int]]) -> None:
    """Выводит статистику по классам."""
    counts: dict[str, int] = {}
    for sample in samples:
        name = str(sample["class_name"])
        counts[name] = counts.get(name, 0) + 1

    print("\n=== Статистика корпуса ===")
    print(f"Всего примеров: {len(samples)}")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")


def main() -> None:
    samples = collect_corpus()
    if len(samples) < MIN_SAMPLES:
        raise RuntimeError(
            f"Собрано только {len(samples)} примеров, требуется минимум {MIN_SAMPLES}."
        )

    save_corpus(samples, OUTPUT_CSV)
    print_stats(samples)
    print(f"\nКорпус сохранён: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
