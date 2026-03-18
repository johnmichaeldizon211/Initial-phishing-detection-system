from __future__ import annotations

from pathlib import Path
import time
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.edge.service import Service as EdgeService

from .utils import ensure_dir


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme:
        return url
    return f"http://{url}"


def capture_screenshot(
    url: str,
    output_path: str | Path,
    driver_path: str | None = None,
    browser: str = "chrome",
    headless: bool = True,
    window_size: tuple[int, int] = (1280, 720),
    timeout: int = 20,
    wait_seconds: float = 2.0,
) -> Path:
    url = _normalize_url(url)
    if not url:
        raise ValueError("URL is empty.")

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    browser = browser.lower().strip()

    if browser == "edge":
        options = webdriver.EdgeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument(f"--window-size={window_size[0]},{window_size[1]}")
        service = EdgeService(driver_path) if driver_path else None
        driver = webdriver.Edge(service=service, options=options)
    else:
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument(f"--window-size={window_size[0]},{window_size[1]}")
        service = ChromeService(driver_path) if driver_path else None
        driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        time.sleep(wait_seconds)
        driver.save_screenshot(str(output_path))
    finally:
        driver.quit()

    return output_path
