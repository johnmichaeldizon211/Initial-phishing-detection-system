from __future__ import annotations
import argparse
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from backend.nlp import train_nlp_model

PHISH_TEMPLATES = [
    "Urgent action required: your {brand} account is locked. Verify here: {url}",
    "Security alert: suspicious login detected on {brand}. Reset now at {url}",
    "Your {brand} payment failed. Update billing info at {url}",
    "We detected unusual activity. Confirm your identity: {url}",
    "Final notice: account will be closed today. Sign in: {url}",
]

LEGIT_TEMPLATES = [
    "Monthly statement for your {brand} account is ready.",
    "Your receipt from {brand} is attached. Thank you for your purchase.",
    "Welcome to {brand}. Your account setup is complete.",
    "Your appointment with {brand} is confirmed for {date}.",
    "Newsletter: updates from {brand} and new features.",
]

BRANDS = ["Metro Bank", "PaySecure", "CloudMail", "ShopWorld", "UniPortal"]
PHISH_URLS = [
    "https://example.com/verify",
    "https://example.com/login",
    "https://example.com/security",
    "https://example.com/account",
]


def _random_date() -> str:
    return random.choice(["April 2", "May 14", "June 9", "July 22"]) + ", 2026"


def generate_emails(count: int) -> pd.DataFrame:
    rows = []
    for _ in range(count):
        template = random.choice(PHISH_TEMPLATES)
        brand = random.choice(BRANDS)
        url = random.choice(PHISH_URLS)
        rows.append({"text": template.format(brand=brand, url=url), "label": 1})

    for _ in range(count):
        template = random.choice(LEGIT_TEMPLATES)
        brand = random.choice(BRANDS)
        date = _random_date()
        rows.append({"text": template.format(brand=brand, date=date), "label": 0})

    random.shuffle(rows)
    return pd.DataFrame(rows)


def _draw_login_image(path: Path, label: str) -> None:
    width, height = 224, 224
    bg = (255, 230, 230) if label == "phishing" else (232, 240, 255)
    header = (30, 75, 138)
    accent = (217, 83, 79) if label == "phishing" else (51, 122, 183)

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.rectangle([0, 0, width, 36], fill=header)
    title = "Secure Login" if label == "legit" else "Verify Account"
    draw.text((10, 10), title, fill=(255, 255, 255), font=font)

    draw.text((18, 60), "Username", fill=(80, 80, 80), font=font)
    draw.rectangle([18, 75, width - 18, 95], outline=(160, 160, 160), width=1)

    draw.text((18, 110), "Password", fill=(80, 80, 80), font=font)
    draw.rectangle([18, 125, width - 18, 145], outline=(160, 160, 160), width=1)

    draw.rectangle([18, 165, width - 18, 190], fill=accent)
    btn_text = "Continue" if label == "legit" else "Confirm"
    draw.text((width // 2 - 25, 172), btn_text, fill=(255, 255, 255), font=font)

    if label == "phishing":
        draw.text((18, 200), "Security alert", fill=(150, 0, 0), font=font)

    img.save(path)


def generate_screenshots(root: Path, per_class: int) -> None:
    phishing_dir = root / "phishing"
    legit_dir = root / "legit"
    phishing_dir.mkdir(parents=True, exist_ok=True)
    legit_dir.mkdir(parents=True, exist_ok=True)

    for i in range(per_class):
        _draw_login_image(phishing_dir / f"phish_{i+1:03d}.png", "phishing")
        _draw_login_image(legit_dir / f"legit_{i+1:03d}.png", "legit")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo data for phishing project.")
    parser.add_argument("--out-dir", default="data", help="Output data directory.")
    parser.add_argument("--emails-per-class", type=int, default=120)
    parser.add_argument("--images-per-class", type=int, default=40)
    parser.add_argument("--train-nlp", action="store_true", help="Train NLP model after generating data.")
    parser.add_argument("--train-vision", action="store_true", help="Train vision model after generating data.")
    parser.add_argument("--vision-weights", default="imagenet", help="Pass weights to vision training.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / args.out_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    emails_df = generate_emails(args.emails_per_class)
    email_path = data_dir / "demo_emails.csv"
    emails_df.to_csv(email_path, index=False)

    screenshots_dir = data_dir / "screenshots"
    generate_screenshots(screenshots_dir, args.images_per_class)

    print(f"Demo emails saved to: {email_path}")
    print(f"Demo screenshots saved to: {screenshots_dir}")

    if args.train_nlp:
        print("Training NLP model...")
        train_nlp_model(email_path, base_dir / "models")
        print("NLP model saved to models/")

    if args.train_vision:
        print("Training vision model...")
        cmd = [
            sys.executable,
            "scripts/vision_train.py",
            "--data-dir",
            str(screenshots_dir),
            "--epochs",
            "3",
            "--backend",
            "auto",
            "--weights",
            args.vision_weights,
        ]
        subprocess.run(cmd, check=True, cwd=base_dir)


if __name__ == "__main__":
    main()
