from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

from backend.capture import capture_screenshot
from backend.combine import combine_scores
from backend.nlp import explain_nlp, load_nlp_model, predict_nlp_proba, train_nlp_model
from backend.utils import extract_urls
from backend.vision import load_vision_model, predict_vision_proba

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
VISION_TF_PATH = MODEL_DIR / "vision_model.keras"
VISION_SK_PATH = MODEL_DIR / "vision_sklearn.joblib"
NLP_THRESHOLD = 0.5
VISION_THRESHOLD = 0.1
COMBINED_THRESHOLD = 0.5


def resolve_vision_model_path() -> Path:
    if VISION_TF_PATH.exists():
        return VISION_TF_PATH
    if VISION_SK_PATH.exists():
        return VISION_SK_PATH
    return VISION_TF_PATH


def is_streamlit_cloud() -> bool:
    if os.environ.get("STREAMLIT_CLOUD"):
        return True
    if Path("/mount/src").exists():
        return True
    if Path("/home/appuser").exists():
        return True
    return False


IS_CLOUD = is_streamlit_cloud()

st.set_page_config(page_title="Phishing Detection System", layout="wide")

def _models_ready() -> bool:
    nlp_ok = (MODEL_DIR / "tfidf.joblib").exists() and (MODEL_DIR / "nlp_model.joblib").exists()
    vision_ok = VISION_TF_PATH.exists() or VISION_SK_PATH.exists()
    return nlp_ok and vision_ok


def _auto_generate_demo_models():
    if not IS_CLOUD:
        return
    if _models_ready():
        return
    if st.session_state.get("auto_demo_done"):
        return

    st.session_state["auto_demo_done"] = True
    try:
        with st.spinner("Setting up demo models for Streamlit Cloud..."):
            subprocess.run([sys.executable, "demo_setup.py"], check=True, cwd=BASE_DIR)

            demo_email_path = BASE_DIR / "data" / "demo_emails.csv"
            train_nlp_model(
                data_path=demo_email_path,
                model_dir=MODEL_DIR,
                text_col="text",
                label_col="label",
            )

            cmd = [
                sys.executable,
                "vision_train.py",
                "--data-dir",
                str(BASE_DIR / "data" / "screenshots"),
                "--backend",
                "sklearn",
                "--epochs",
                "3",
            ]
            subprocess.run(cmd, check=True, cwd=BASE_DIR)
        st.success("Demo models ready.")
    except Exception as exc:
        st.error(f"Auto demo setup failed: {exc}")


_auto_generate_demo_models()

CUSTOM_CSS = """
<style>
:root {
  --blue-dark: #1e4b8a;
  --blue: #337ab7;
  --blue-deep: #245680;
  --bg: #f4f7fc;
  --text: #334e68;
  --border: #dee2e6;
  --danger: #d9534f;
  --success: #5cb85c;
}

html, body, [class*="css"] {
  font-family: 'Segoe UI', Tahoma, sans-serif;
}

.stApp {
  background-color: #e9effb;
}

.block-container {
  padding: 1.25rem 5% 2rem;
  max-width: 1200px;
}

.header-bar {
  background: var(--blue-dark);
  color: white;
  padding: 26px 0 22px;
  margin: 0 -5% 28px;
  position: relative;
  z-index: 2;
}

.header-inner {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 5%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  min-height: 52px;
}

.header-title {
  font-size: 1.8rem;
  font-weight: 500;
  line-height: 1.25;
  padding-top: 4px;
  display: block;
}

.shield-icon {
  line-height: 1;
  display: inline-flex;
  align-items: center;
}

.shield-icon {
  font-size: 2.2rem;
}

.section-title {
  font-size: 1.2rem;
  font-weight: 600;
  color: var(--text);
  margin: 0 0 12px;
}

.input-hint {
  text-align: center;
  margin-bottom: 16px;
}

div[data-testid="stTextArea"] textarea {
  background: white;
  border: 1px solid #ced4da;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  font-size: 0.95rem;
}

div[data-testid="stTextInput"] input {
  background: white;
  border: 1px solid #ced4da;
  border-radius: 6px;
  padding: 10px 12px;
}

div[data-testid="stFileUploader"] section {
  background: white;
  border: 1px dashed #ced4da;
  border-radius: 6px;
}

.stButton > button {
  background: var(--blue);
  color: white;
  border: none;
  padding: 12px 60px;
  border-radius: 4px;
  font-size: 1.05rem;
  font-weight: 700;
  box-shadow: 0 4px 0 var(--blue-deep);
}

.stButton > button:hover {
  background: #2d6da3;
}

section[data-testid="stSidebar"] {
  background: #f3f6fc;
  border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .block-container {
  padding-top: 1.5rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
  color: #1f2937;
}

section[data-testid="stSidebar"] .stSlider > div {
  padding-top: 6px;
}

section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
  background: var(--blue);
}

section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div > div {
  background: var(--blue);
  box-shadow: 0 0 0 2px rgba(51, 122, 183, 0.2);
}

section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
  background: white;
  border: 2px solid var(--blue);
}

section[data-testid="stSidebar"] .stCheckbox > label,
section[data-testid="stSidebar"] .stSelectbox > label,
section[data-testid="stSidebar"] .stTextInput > label {
  font-weight: 600;
}

section[data-testid="stSidebar"] div[data-testid="stTextInput"] input,
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div {
  background: white;
  border: 1px solid #ced4da;
  border-radius: 6px;
}

.card {
  background: white;
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
  box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.card h3 {
  background: #f8f9fa;
  padding: 12px;
  font-size: 1.1rem;
  color: var(--text);
  border-bottom: 1px solid var(--border);
  text-align: center;
  margin: 0;
}

.card-inner {
  padding: 20px;
}

.score-badge {
  padding: 15px;
  border-radius: 6px;
  color: white;
  font-weight: bold;
  font-size: 1.3rem;
  margin-bottom: 15px;
  text-align: center;
}

.score-good {
  background: var(--success);
  border-bottom: 5px solid #4cae4c;
}

.score-bad {
  background: var(--danger);
  border-bottom: 5px solid #d43f3a;
}

.score-muted {
  background: #9aa6b2;
  border-bottom: 5px solid #8b96a1;
}

.status-text {
  text-align: center;
  color: #333;
  font-weight: 500;
}

.screenshot-preview {
  border: 1px solid #ddd;
  background: #f1f4f9;
  margin-bottom: 15px;
}

.screenshot-img {
  display: block;
  width: 100%;
  height: auto;
}

.mock-browser-header {
  background: #234e82;
  color: white;
  font-size: 0.8rem;
  padding: 8px;
  text-align: center;
}

.mock-form {
  padding: 15px;
  background: white;
}

.mock-label {
  display: block;
  text-align: left;
  font-size: 0.75rem;
  color: #666;
  margin-top: 8px;
  margin-bottom: 2px;
}

.input-line {
  border: 1px solid #ccc;
  height: 25px;
  width: 100%;
  margin-bottom: 5px;
}

.mock-button {
  background: var(--blue);
  color: white;
  font-size: 0.8rem;
  padding: 8px;
  margin-top: 15px;
  text-align: center;
}

.warning-text {
  color: var(--danger);
  font-weight: bold;
  text-align: center;
  font-size: 0.9rem;
}

.final-result {
  background: white;
  border: 1px solid var(--border);
  border-top: 5px solid var(--danger);
  border-radius: 8px;
  display: flex;
  align-items: center;
  padding: 25px 40px;
  box-shadow: 0 4px 15px rgba(0,0,0,0.1);
  gap: 24px;
}

.shield-circle {
  width: 80px;
  height: 80px;
  background: var(--danger);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 2.4rem;
  box-shadow: 0 0 0 10px rgba(217, 83, 79, 0.1);
}

.alert-title {
  color: var(--danger);
  font-size: 1.8rem;
  font-weight: bold;
  margin: 0 0 4px;
}

.alert-details p {
  color: #555;
  font-size: 1.1rem;
  margin: 0;
}

.combined-score {
  background: #1c334d;
  color: white;
  padding: 15px 30px;
  border-radius: 4px;
  font-size: 1.1rem;
  margin-left: auto;
}

.combined-score strong {
  font-size: 1.5rem;
}

@media (max-width: 900px) {
  .final-result {
    flex-direction: column;
    text-align: center;
  }
  .combined-score {
    margin-left: 0;
    width: 100%;
    text-align: center;
  }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="header-bar">
      <div class="header-inner">
        <span class="shield-icon">&#x1F6E1;&#xFE0F;</span>
        <div class="header-title">Phishing Detection System</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    weight_nlp = st.slider("Weight: NLP", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    weight_vision = st.slider("Weight: Vision", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    vision_threshold = st.slider(
        "Vision threshold",
        min_value=0.0,
        max_value=1.0,
        value=VISION_THRESHOLD,
        step=0.05,
    )
    st.caption("Weights are normalized automatically.")

    use_selenium = st.checkbox(
        "Capture screenshot from URL (Selenium)",
        value=False if IS_CLOUD else True,
    )
    if IS_CLOUD and use_selenium:
        st.warning("Selenium may fail on Streamlit Cloud. Upload a screenshot if capture fails.")
    browser = st.selectbox("Browser", ["chrome", "edge"], index=0)
    driver_path = st.text_input("WebDriver path (optional)", value="")

    st.subheader("Model Status")
    nlp_ready = (MODEL_DIR / "tfidf.joblib").exists() and (MODEL_DIR / "nlp_model.joblib").exists()
    vision_ready = VISION_TF_PATH.exists() or VISION_SK_PATH.exists()
    if nlp_ready:
        st.success("NLP model ready")
    else:
        st.warning("NLP model missing")
    if vision_ready:
        st.success("Vision model ready")
    else:
        st.warning("Vision model missing")

    st.subheader("Utilities")
    demo_btn = st.button("Generate Demo Models")
    nlp_data_path = st.text_input("NLP dataset path", value="data/demo_emails.csv")
    nlp_text_col = st.text_input("NLP text column", value="text")
    nlp_label_col = st.text_input("NLP label column", value="label")
    train_nlp_btn = st.button("Train NLP Model")

    vision_data_path = st.text_input("Vision dataset path", value="data/screenshots")
    vision_epochs = st.number_input("Vision epochs", min_value=1, max_value=20, value=3, step=1)
    vision_backend = st.selectbox(
        "Vision backend",
        ["auto", "tensorflow", "sklearn"],
        index=2 if IS_CLOUD else 0,
    )
    vision_weights = st.selectbox("Vision weights", ["imagenet", "none"], index=1)
    train_vision_btn = st.button("Train Vision Model")

    if vision_backend == "tensorflow":
        try:
            import tensorflow as _  # noqa: F401
        except Exception:
            st.warning("TensorFlow not available. Use sklearn backend or Python 3.11/3.12.")

st.markdown('<div class="input-hint">Paste the suspicious email text below and click Analyze.</div>', unsafe_allow_html=True)

email_text = st.text_area(
    "Email content",
    height=200,
    value=(
        "Dear Customer,\n"
        "Please urgent! Your account has been compromised. Click the link below to secure your account:\n"
        "http://fakebank-login.com/verify"
    ),
)

detected_urls = extract_urls(email_text)
auto_url = detected_urls[0] if detected_urls else ""
if auto_url:
    st.caption(f"Detected URL from email: {auto_url}")

url_input = st.text_input("URL to inspect (optional)", placeholder="https://example.com/login")
image_upload = st.file_uploader("Or upload a screenshot image", type=["png", "jpg", "jpeg"])

analyze = st.button("Analyze")

nlp_score = None
vision_score = None
screenshot_path = None
nlp_vectorizer = None
nlp_model = None
vision_backend_used = None


@st.cache_resource(show_spinner=False)
def get_nlp_components(model_dir: Path):
    return load_nlp_model(model_dir)


@st.cache_resource(show_spinner=False)
def get_vision_component(model_path: Path):
    return load_vision_model(model_path)


def _image_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def _score_badge(
    label: str,
    score: float | None,
    threshold: float = 0.5,
    enabled: bool = True,
) -> str:
    if score is None:
        return f'<div class="score-badge score-good">{label}: --</div>'
    if not enabled:
        return f'<div class="score-badge score-muted">{label}: {score * 100:.1f}% (ignored)</div>'
    css_class = "score-bad" if score >= threshold else "score-good"
    return f'<div class="score-badge {css_class}">{label}: {score * 100:.1f}%</div>'


def _render_nlp_card(score: float | None, threshold: float, enabled: bool) -> str:
    if score is None:
        status = "NLP score unavailable"
    elif not enabled:
        status = "NLP weight is 0 (ignored)"
    else:
        status = "Suspicious language detected" if (score or 0) >= threshold else "Language looks safe"
    badge = _score_badge("Phishing Score", score, threshold, enabled)
    return f"""
    <div class="card">
      <h3>NLP Analysis</h3>
      <div class="card-inner">
        {badge}
        <div class="status-text">{status}</div>
      </div>
    </div>
    """


def _render_screenshot_card(
    image_b64: str | None,
    score: float | None,
    threshold: float,
    enabled: bool,
) -> str:
    if image_b64:
        preview = f'<img class="screenshot-img" src="data:image/png;base64,{image_b64}" />'
        if not enabled:
            warning = "Vision weight is 0 (ignored)"
        else:
            warning = "Potential phishing site" if (score or 0) >= threshold else "Site looks safe"
    else:
        preview = """
        <div class="mock-browser-header">Login page preview</div>
        <div class="mock-form">
          <label class="mock-label">User name</label>
          <div class="input-line"></div>
          <label class="mock-label">Password</label>
          <div class="input-line"></div>
          <div class="mock-button">Submit</div>
        </div>
        """
        warning = "No screenshot available"

    return f"""
    <div class="card">
      <h3>Website Screenshot</h3>
      <div class="card-inner">
        <div class="screenshot-preview">{preview}</div>
        <div class="warning-text">{warning}</div>
      </div>
    </div>
    """


def _render_vision_card(score: float | None, threshold: float, enabled: bool) -> str:
    if score is None:
        status = "Vision score unavailable"
    elif not enabled:
        status = "Vision weight is 0 (ignored)"
    else:
        status = "Fake page elements found" if (score or 0) >= threshold else "Site looks safe"
    badge = _score_badge("Website Risk", score, threshold, enabled)
    return f"""
    <div class="card">
      <h3>Vision Analysis</h3>
      <div class="card-inner">
        {badge}
        <div class="status-text">{status}</div>
      </div>
    </div>
    """


def _render_final_banner(score: float | None) -> str:
    if score is None:
        title = "No verdict"
        detail = "Run NLP and/or vision analysis to get a combined score."
        combined = "--"
    else:
        title = "Phishing detected!" if score >= COMBINED_THRESHOLD else "Likely safe"
        detail = "High risk of phishing attack" if score >= COMBINED_THRESHOLD else "Low risk detected"
        combined = f"{score * 100:.1f}%"

    return f"""
    <div class="final-result">
      <div class="shield-circle">&#x1F6E1;&#xFE0F;</div>
      <div class="alert-details">
        <div class="alert-title">{title}</div>
        <p>{detail}</p>
      </div>
      <div class="combined-score">Combined Score: <strong>{combined}</strong></div>
    </div>
    """

if demo_btn:
    try:
        with st.spinner("Generating demo data and models..."):
            subprocess.run([sys.executable, "demo_setup.py"], check=True, cwd=BASE_DIR)

            demo_email_path = BASE_DIR / "data" / "demo_emails.csv"
            train_nlp_model(
                data_path=demo_email_path,
                model_dir=MODEL_DIR,
                text_col="text",
                label_col="label",
            )

            cmd = [
                sys.executable,
                "vision_train.py",
                "--data-dir",
                str(BASE_DIR / "data" / "screenshots"),
                "--backend",
                "sklearn",
                "--epochs",
                "3",
            ]
            subprocess.run(cmd, check=True, cwd=BASE_DIR)

        get_nlp_components.clear()
        get_vision_component.clear()
        st.success("Demo models generated in models/")
    except Exception as exc:
        st.error(f"Demo setup failed: {exc}")

if train_nlp_btn:
    data_path = (BASE_DIR / nlp_data_path).resolve()
    if not data_path.exists():
        st.error(f"NLP dataset not found: {data_path}")
    else:
        try:
            with st.spinner("Training NLP model..."):
                train_nlp_model(
                    data_path=data_path,
                    model_dir=MODEL_DIR,
                    text_col=nlp_text_col or None,
                    label_col=nlp_label_col or None,
                )
            get_nlp_components.clear()
            st.success("NLP model trained and saved to models/")
        except Exception as exc:
            st.error(f"NLP training failed: {exc}")

if train_vision_btn:
    data_path = (BASE_DIR / vision_data_path).resolve()
    if not data_path.exists():
        st.error(f"Vision dataset not found: {data_path}")
    else:
        try:
            with st.spinner("Training vision model..."):
                cmd = [
                    sys.executable,
                    "vision_train.py",
                    "--data-dir",
                    str(data_path),
                    "--epochs",
                    str(int(vision_epochs)),
                    "--backend",
                    vision_backend,
                    "--weights",
                    vision_weights,
                ]
                subprocess.run(cmd, check=True, cwd=BASE_DIR)
            get_vision_component.clear()
            st.success("Vision model trained and saved to models/")
        except Exception as exc:
            st.error(f"Vision training failed: {exc}")

if analyze:
    if not email_text.strip():
        st.warning("Email text is required for NLP analysis.")
    else:
        try:
            nlp_vectorizer, nlp_model = get_nlp_components(MODEL_DIR)
            nlp_score = predict_nlp_proba(email_text, nlp_vectorizer, nlp_model)
        except Exception as exc:
            st.error(f"Failed to load NLP model. Train it first. Error: {exc}")

    if image_upload is not None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        screenshot_path = OUTPUT_DIR / f"upload_{timestamp}.png"
        with open(screenshot_path, "wb") as f:
            f.write(image_upload.getbuffer())
    elif (url_input.strip() or auto_url) and use_selenium:
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            url_to_use = url_input.strip() or auto_url
            with st.spinner("Capturing screenshot..."):
                temp_path = OUTPUT_DIR / f"capture_{timestamp}.png"
                capture_screenshot(
                    url_to_use,
                    temp_path,
                    driver_path=driver_path or None,
                    browser=browser,
                    headless=True,
                )
            screenshot_path = temp_path
        except Exception as exc:
            st.error(f"Screenshot capture failed: {exc}")

    if screenshot_path is not None:
        try:
            model_path = resolve_vision_model_path()
            if not model_path.exists():
                raise FileNotFoundError("Vision model not found. Train it first.")

            vision_model = get_vision_component(model_path)
            if isinstance(vision_model, tuple) and len(vision_model) == 2:
                vision_backend_used = vision_model[0]
            vision_score = predict_vision_proba(screenshot_path, vision_model)
        except Exception as exc:
            st.error(f"Vision analysis failed: {exc}")

    combined_score = combine_scores(nlp_score, vision_score, weight_nlp, weight_vision)
    nlp_enabled = weight_nlp > 0
    vision_enabled = weight_vision > 0
    terms_list: list[dict] = []

    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    image_b64 = _image_to_base64(screenshot_path) if screenshot_path else None
    if image_b64 == "":
        image_b64 = None

    with col1:
        st.markdown(_render_nlp_card(nlp_score, NLP_THRESHOLD, nlp_enabled), unsafe_allow_html=True)
    with col2:
        st.markdown(
            _render_screenshot_card(image_b64, vision_score, vision_threshold, vision_enabled),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            _render_vision_card(vision_score, vision_threshold, vision_enabled),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Final Verdict</div>', unsafe_allow_html=True)
    st.markdown(_render_final_banner(combined_score), unsafe_allow_html=True)

    if nlp_score is not None and nlp_vectorizer is not None and nlp_model is not None:
        terms = explain_nlp(email_text, nlp_vectorizer, nlp_model)
        if terms:
            st.markdown('<div class="section-title">Top NLP Signals</div>', unsafe_allow_html=True)
            for term, weight in terms:
                st.write(f"{term} (weight {weight:.3f})")
            terms_list = [{"term": term, "weight": weight} for term, weight in terms]

    report = {
        "nlp_score": nlp_score,
        "vision_score": vision_score,
        "combined_score": combined_score,
        "verdict": "phishing" if (combined_score or 0) >= COMBINED_THRESHOLD else "likely_safe",
        "url_used": url_input.strip() or auto_url or None,
        "screenshot_path": str(screenshot_path) if screenshot_path else None,
        "top_nlp_terms": terms_list,
        "weights": {"nlp": weight_nlp, "vision": weight_vision},
        "vision_backend": vision_backend_used,
        "thresholds": {
            "nlp": NLP_THRESHOLD,
            "vision": vision_threshold,
            "combined": COMBINED_THRESHOLD,
        },
    }

    st.download_button(
        "Download Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="phishing_report.json",
        mime="application/json",
    )
