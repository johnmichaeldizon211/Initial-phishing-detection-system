from __future__ import annotations

from pathlib import Path
import re
from bs4 import BeautifulSoup

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_email_text(text: str | None) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Strip HTML while keeping visible text.
    soup = BeautifulSoup(text, "lxml")
    text = soup.get_text(" ")

    text = URL_RE.sub(" <url> ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def extract_urls(text: str | None) -> list[str]:
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    return URL_RE.findall(text)
