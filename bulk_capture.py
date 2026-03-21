from __future__ import annotations

import argparse
import time
from pathlib import Path

from backend.capture import capture_screenshot
from backend.utils import ensure_dir


def _read_urls(path: Path) -> list[str]:
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk capture screenshots for legit sites.")
    parser.add_argument("--urls", required=True, help="Text file with one URL per line.")
    parser.add_argument("--out-dir", default="data/screenshots/legit")
    parser.add_argument("--browser", default="chrome", choices=["chrome", "edge"])
    parser.add_argument("--driver-path", default="", help="Optional webdriver path.")
    parser.add_argument("--headless", action="store_true", help="Run browser headless.")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--wait", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of URLs (0 = no limit).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if output file exists.")
    args = parser.parse_args()

    url_path = Path(args.urls)
    if not url_path.exists():
        raise FileNotFoundError(f"URL list not found: {url_path}")

    urls = _read_urls(url_path)
    if args.limit and args.limit > 0:
        urls = urls[: args.limit]

    out_dir = ensure_dir(Path(args.out_dir))
    driver_path = args.driver_path or None

    ok = 0
    fail = 0
    for idx, url in enumerate(urls, start=1):
        out_path = out_dir / f"legit_{idx:03d}.png"
        if args.skip_existing and out_path.exists():
            print(f"[skip] {out_path.name}")
            continue
        try:
            capture_screenshot(
                url,
                out_path,
                driver_path=driver_path,
                browser=args.browser,
                headless=args.headless,
                timeout=args.timeout,
                wait_seconds=args.wait,
            )
            ok += 1
            print(f"[ok] {url} -> {out_path.name}")
        except Exception as exc:
            fail += 1
            print(f"[fail] {url} -> {exc}")
        time.sleep(0.5)

    print(f"Done. Success: {ok}, Failed: {fail}, Total: {len(urls)}")


if __name__ == "__main__":
    main()
