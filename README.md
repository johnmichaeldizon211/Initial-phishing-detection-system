# Phishing Detection System

A multi-modal phishing detection system that combines:
- **NLP** (TF-IDF + Logistic Regression) for email text
- **Vision** (TensorFlow MobileNetV2 or sklearn fallback) for website screenshots
- **Fusion** (weighted score) to produce a final phishing confidence

## Features
- Streamlit web app UI
- Email text analysis + URL extraction
- Screenshot capture with Selenium (optional)
- Vision model auto-fallback when TensorFlow is unavailable
- JSON report export

## Quick Start (Python 3.14+)
TensorFlow is not available on Python 3.14, so the vision model uses the **sklearn** backend.

```bash
pip install -r requirements.txt
python demo_setup.py
python nlp_train.py --data data/demo_emails.csv --text-col text --label-col label
python vision_train.py --data-dir data/screenshots --backend sklearn --epochs 3
streamlit run app.py
```

## Full CNN (Python 3.11/3.12)
Use Python 3.11/3.12 if you want MobileNetV2 (TensorFlow):

```bash
pip install -r requirements.txt
python vision_train.py --data-dir data/screenshots --backend tensorflow --epochs 5
```

## Streamlit Deploy
1. Push this repo to GitHub
2. Go to Streamlit Community Cloud → New app
3. Select `app.py`
4. `runtime.txt` pins Python 3.11 for TensorFlow compatibility
5. `packages.txt` installs system libs used by vision dependencies

## Notes
- `data/`, `models/`, and `outputs/` are ignored in git.
- Use a VM/sandbox for Selenium URL capture.
