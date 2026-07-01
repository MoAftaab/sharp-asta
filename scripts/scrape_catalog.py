from __future__ import annotations

import json
import re
import time
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT = Path("data/catalog_scraped.json")
TEST_TYPE_CODES = {"A", "B", "C", "D", "E", "K", "P", "S"}


def absolute_url(href: str | None) -> str | None:
    if not href:
        return None
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return f"https://www.shl.com{href}"
    return None


def extract_codes(text: str) -> list[str]:
    found = []
    for token in re.findall(r"\b[A-Z]\b", text):
        if token in TEST_TYPE_CODES and token not in found:
            found.append(token)
    return found


def scrape_listing(page, start: int) -> list[dict]:
    page.goto(f"{BASE_URL}?start={start}&type=1&sortdir=asc", wait_until="networkidle", timeout=60000)
    time.sleep(1)
    soup = BeautifulSoup(page.content(), "html.parser")
    rows = soup.select("tr, .product-catalogue__item, .custom__table-row")
    items = []

    for row in rows:
        link = row.select_one("a[href*='/solutions/products/']")
        if not link:
            continue
        name = link.get_text(" ", strip=True)
        url = absolute_url(link.get("href"))
        if not name or not url or "product-catalog" in url:
            continue
        text = row.get_text(" ", strip=True)
        codes = extract_codes(text)
        items.append(
            {
                "name": name,
                "url": url,
                "test_types": codes,
                "remote_testing": "remote" in text.lower() or "yes" in text.lower(),
                "adaptive_irt": "adaptive" in text.lower() or "irt" in text.lower(),
                "description": "",
                "duration_minutes": None,
                "languages": ["English"],
                "job_levels": [],
                "keywords": [],
            }
        )
    return items


def scrape_detail(page, item: dict) -> dict:
    try:
        page.goto(item["url"], wait_until="networkidle", timeout=30000)
        time.sleep(0.5)
        soup = BeautifulSoup(page.content(), "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.select("p")]
        item["description"] = next((p for p in paragraphs if len(p) > 60), "")[:800]
        duration_text = soup.get_text(" ", strip=True)
        match = re.search(r"(\d{1,3})\s*(?:minutes|minute|min)", duration_text, re.I)
        if match:
            item["duration_minutes"] = int(match.group(1))
    except Exception as exc:
        print(f"Detail scrape failed for {item['url']}: {exc}")
    return item


def main() -> None:
    OUTPUT.parent.mkdir(exist_ok=True)
    seen = set()
    catalog = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        start = 0
        while True:
            print(f"Scraping start={start}")
            items = scrape_listing(page, start)
            fresh = [item for item in items if item["url"] not in seen]
            if not fresh:
                break
            for item in fresh:
                seen.add(item["url"])
                catalog.append(scrape_detail(page, item))
            start += 12
        browser.close()

    OUTPUT.write_text(json.dumps(catalog, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved {len(catalog)} items to {OUTPUT}")


if __name__ == "__main__":
    main()

