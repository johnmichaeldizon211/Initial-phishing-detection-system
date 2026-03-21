# Backend Setup

This backend provides:
- NLP model (TF-IDF + Logistic Regression)
- Vision model (MobileNetV2)
- Score fusion
- Streamlit web app

## 1. Install dependencies

```
pip install -r requirements.txt
```

Notes:
- On Python 3.13+, TensorFlow is skipped automatically. The vision model will use the sklearn backend.
- For TensorFlow (MobileNetV2), use Python 3.11 or 3.12.

## 2. Train the NLP model

Place your email dataset CSV in `data/` and run:

```
python nlp_train.py --data data/emails.csv --text-col text --label-col label
```

If your dataset uses different column names, set `--text-col` and `--label-col` accordingly.

Outputs:
- `models/tfidf.joblib`
- `models/nlp_model.joblib`
- `models/nlp_report.json`

## 2B. Train BERT-tiny (optional, better accuracy)

Install the extra dependencies:

```
pip install -r requirements-bert.txt
```

Then run:

```
python bert_train.py --data data/emails.csv --text-col "Email Text" --label-col "Email Type"
```

Outputs:
- `models/bert_tiny/`
- `models/bert_tiny/bert_report.json`

## 3. Train the vision model

Prepare screenshots in this structure:

```
data/screenshots/
  phishing/
  legit/
```

Then run:

```
python vision_train.py --data-dir data/screenshots --epochs 5
```

Outputs:
- `models/vision_model.keras`
- `models/vision_model.json`

If TensorFlow is not available, the sklearn fallback will save:
- `models/vision_sklearn.joblib`
- `models/vision_sklearn.json`

## 4. Run the Streamlit app

```
streamlit run app.py
```

## 5. Evaluation (Required for report)

NLP evaluation:

```
python eval_nlp.py --data data/emails.csv --text-col "Email Text" --label-col "Email Type"
```

Vision evaluation (requires screenshots dataset):

```
python eval_vision.py --data-dir data/screenshots --model-path models/vision_model.keras
```

Combined evaluation (requires paired CSV with email + screenshot path):

```
python eval_combined.py --pairs data/pairs.csv --weight-nlp 0.5 --weight-vision 0.5
```

Outputs are saved to `outputs/`:
- `nlp_eval.json`, `nlp_confusion.csv`, `nlp_misclassified.csv`
- `vision_eval.json`, `vision_confusion.csv`, `vision_misclassified.csv`
- `combined_eval.json`, `combined_confusion.csv`

## Quick Demo (No Kaggle needed)

Generate a small synthetic dataset and (optionally) train models:

```
python demo_setup.py
python nlp_train.py --data data/demo_emails.csv --text-col text --label-col label
python vision_train.py --data-dir data/screenshots --epochs 3 --backend auto --weights none
```

Notes:
- Use `--weights none` if you do not want to download ImageNet weights.
- Demo data is only for showcasing the pipeline, not real accuracy.

## 5. Selenium notes

To capture live screenshots, install Chrome or Edge WebDriver and set the path in the app sidebar.
For safety, run on a VM or sandbox environment.
