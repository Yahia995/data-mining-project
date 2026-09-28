# 🧠 Arabic Emotion Detection Pipeline

> End-to-end data pipeline for emotion detection in Arabic text (Algerian Darija), from data acquisition through ML evaluation to a lightweight API and dashboard.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Research Question](#-research-question)
- [Dataset Description](#-dataset-description)
- [Methodology Overview](#-methodology-overview)
- [Architecture](#-architecture)
- [Results Summary](#-results-summary)
- [Setup Instructions](#-setup-instructions)
- [Project Structure](#-project-structure)
- [Running the Pipeline](#-running-the-pipeline)
- [Starting the API & Dashboard](#-starting-the-api--dashboard)
- [Running Tests](#-running-tests)
- [Docker Instructions](#-docker-instructions)
- [Team Members](#-team-members)
- [License](#-license)

---

## 🔬 Research Question

> *Can we build a reproducible data pipeline that automatically classifies emotions in Arabic social-media text, and can we analyze how emotional patterns differ across emotion categories, identifying which emotions are reliably detectable and which remain ambiguous?*

---

## 📊 Dataset Description

### Primary Dataset — Dz-Emotion

| Property       | Detail                                              |
|----------------|-----------------------------------------------------|
| **Source**      | [HuggingFace Hub](https://huggingface.co/datasets/Houdna-khilouf/Dz-Emotion) (`Houdna-khilouf/Dz-Emotion`) |
| **Authors**     | Houdna Khilouf et al.                               |
| **Language**    | Algerian Darija (Arabic dialect)                     |
| **Size**        | 6,000 labeled samples                                |
| **Emotions**    | 6 Ekman emotions — Anger, Disgust, Fear, Happiness, Sadness, Surprise |
| **Balance**     | Perfectly balanced (1,000 samples per class)         |
| **License**     | Academic / research use (no explicit open-source license on HuggingFace) |

> **Note:** The raw dataset is **not** committed to the repository. It is downloaded programmatically via the ingestion script.

### Supplementary Resource — ArSEL Lexicon

- **Source:** [GitHub — Husseinjd/ArSEL](https://github.com/Husseinjd/ArSEL)
- **Description:** 32,000+ Arabic lemmas scored on 8 Plutchik emotions (anger, anticipation, disgust, fear, joy, sadness, surprise, trust)
- **Usage:** Lexicon-based emotion feature engineering

### Optional Evaluation — SemEval-2018 Task 1 (E-c Arabic)

- **Source:** CodaLab / HuggingFace
- **Usage:** Cross-dataset validation (optional, P3)

---

## 🧪 Methodology Overview

The project follows a rigorous, gate-driven data mining methodology:

1. **Problem Definition** — Formalize the research question and define measurable sub-questions
2. **Data Acquisition** — Automated download from HuggingFace via reproducible ingestion script
3. **Data Validation** — Validate raw data against a formal data contract (completeness, validity, uniqueness, format)
4. **Data Quality Assessment** — Produce a comprehensive quality report covering 10 dimensions
5. **Preprocessing** — Arabic-specific text normalization (diacritics removal, tatweel, character variant normalization, stopword removal)
6. **Feature Engineering** — TF-IDF vectorization, ArSEL lexicon emotion scores, stylistic features
7. **Data Splitting** — Stratified 70/15/15 train/validation/test split
8. **Model Training** — Multiple classifiers: Majority Baseline → Logistic Regression → Naive Bayes → Linear SVM → Random Forest
9. **Evaluation** — Accuracy, macro Precision/Recall/F1, per-class metrics, confusion matrices
10. **Error Analysis** — Systematic investigation of misclassifications and emotion ambiguity
11. **Deployment** — FastAPI prediction API + Streamlit dashboard

### Quality Gates

| Gate | Name                        | Purpose                                  |
|------|-----------------------------|------------------------------------------|
| G2   | Data Quality Gate           | Verify dataset passes quality thresholds |
| G3   | Pipeline Reproducibility    | Ensure `run_pipeline.py` reproduces results |
| G5   | Mining Results Gate         | Validate model results are interpretable |
| G6   | Product Readiness Gate      | API and dashboard expose pipeline correctly |
| G7   | Final Reproducibility Test  | Full clean-clone reproducibility check   |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE (run_pipeline.py)              │
│                                                                 │
│  ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌───────────┐  │
│  │ Ingestion│──▶│ Validation │──▶│ Cleaning │──▶│ Splitting │  │
│  │(download)│   │(data       │   │(normalize│   │(stratified│  │
│  │          │   │ contract)  │   │ Arabic)  │   │ 70/15/15) │  │
│  └──────────┘   └────────────┘   └──────────┘   └─────┬─────┘  │
│                                                       │         │
│  ┌──────────────────────────────────────────────────┐  │         │
│  │              Feature Engineering                 │◀─┘         │
│  │  ┌─────────┐  ┌───────────┐  ┌───────────────┐  │           │
│  │  │ TF-IDF  │  │  ArSEL    │  │  Stylistic    │  │           │
│  │  │Features │  │  Lexicon  │  │  Features     │  │           │
│  │  └─────────┘  └───────────┘  └───────────────┘  │           │
│  └──────────────────────┬───────────────────────────┘           │
│                         │                                       │
│  ┌──────────────────────▼───────────────────────────┐           │
│  │                Model Training                    │           │
│  │  Baseline → LogReg → NB → SVM → Random Forest   │           │
│  └──────────────────────┬───────────────────────────┘           │
│                         │                                       │
│  ┌──────────────────────▼───────────────────────────┐           │
│  │                Evaluation                        │           │
│  │  Metrics · Confusion Matrix · Error Analysis     │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                                 ▼
┌─────────────────┐              ┌──────────────────┐
│   FastAPI        │              │   Streamlit      │
│   /predict       │◀────────────│   Dashboard      │
│   /analyze       │              │   (Prediction +  │
│   /health        │              │    Results)      │
└─────────────────┘              └──────────────────┘
```

---

## 📈 Results Summary

> **⚠️ Results will be populated once model training and evaluation are complete (Cycle 3).**

| Model               | Accuracy | Macro F1 | Notes                     |
|----------------------|----------|----------|---------------------------|
| Majority Baseline    | ~16.7%   | —        | Lower bound (balanced)    |
| Logistic Regression  | TBD      | TBD      | TF-IDF features           |
| Naive Bayes          | TBD      | TBD      | TF-IDF features           |
| Linear SVM           | TBD      | TBD      | TF-IDF features           |
| Random Forest        | TBD      | TBD      | TF-IDF + lexicon + style  |

> The best model will be selected based on **macro F1-score** and deployed via the API.

---

## ⚙ Setup Instructions

### Prerequisites

- **Python** 3.11+
- **pip** (or a virtualenv manager)
- **Git**
- **Docker** & **Docker Compose** *(optional, for containerized deployment)*

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Yahia995/data-mining-project.git
cd data-mining-project

# 2. Create and activate a virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
data-mining-project/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies (pinned)
├── run_pipeline.py              # Single-command pipeline orchestrator
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Multi-service orchestration
│
├── docs/                        # Documentation
│   ├── architecture/            # Architecture diagrams & decisions
│   ├── methodology/             # Data mining methodology write-up
│   ├── experiments/             # Experiment log (CSV + notes)
│   └── reports/                 # Data quality report, gate reviews
│
├── data/                        # Data directory (gitignored)
│   ├── raw/                     # Raw downloaded dataset
│   ├── processed/               # Cleaned & split data
│   └── external/                # ArSEL lexicon, external resources
│
├── notebooks/                   # Jupyter notebooks
│   ├── 00-data-inspection.ipynb
│   ├── 01-eda.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-feature-engineering.ipynb
│   ├── 04-modeling.ipynb
│   ├── 05-model-comparison.ipynb
│   └── 06-error-analysis.ipynb
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── config.py                # Central configuration
│   ├── ingestion/               # Dataset download (HuggingFace)
│   ├── preprocessing/           # Arabic normalization & cleaning
│   ├── validation/              # Data contract validation
│   ├── features/                # Feature engineering (TF-IDF, lexicon, stylistic)
│   ├── mining/                  # Model training & prediction
│   └── evaluation/              # Metrics, confusion matrix, error analysis
│
├── api/                         # FastAPI application
│   ├── __init__.py
│   ├── main.py                  # API endpoints (/predict, /analyze, /health)
│   └── schemas.py               # Pydantic request/response models
│
├── dashboard/                   # Streamlit dashboard
│   └── app.py                   # Multi-page dashboard (prediction + results)
│
├── models/                      # Trained model artifacts (gitignored)
│
├── tests/                       # Test suite
│   ├── test_preprocessing.py    # Unit tests for Arabic normalization
│   └── test_api.py              # Integration tests for API endpoints
│
└── .github/
    └── workflows/
        └── ci.yml               # GitHub Actions CI pipeline
```

---

## 🚀 Running the Pipeline

The entire data pipeline — from download to feature extraction — runs with a single command:

```bash
python run_pipeline.py
```

This will execute, in order:

1. **Ingestion** — Download Dz-Emotion from HuggingFace → `data/raw/`
2. **Validation** — Validate raw data against the data contract
3. **Cleaning** — Arabic text normalization → `data/processed/dz_emotion_clean.csv`
4. **Splitting** — Stratified 70/15/15 split → `data/processed/{train,val,test}.csv`
5. **Feature Engineering** — TF-IDF + ArSEL lexicon + stylistic features

> **Reproducibility:** Running this command from a clean clone produces identical results. All random seeds are fixed.

---

## 🌐 Starting the API & Dashboard

### FastAPI (Prediction API)

```bash
# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# API will be available at:
# - Prediction:     POST http://localhost:8000/predict
# - Batch Analysis: POST http://localhost:8000/analyze
# - Health Check:   GET  http://localhost:8000/health
# - API Docs:       GET  http://localhost:8000/docs
```

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "أنا فرحان بزاف اليوم"}'
```

### Streamlit Dashboard

```bash
# Start the dashboard
streamlit run dashboard/app.py --server.port 8501

# Dashboard will be available at http://localhost:8501
```

The dashboard includes:
- **Prediction Page** — Input Arabic text, see predicted emotion with confidence
- **Results Page** — Model comparison, confusion matrix, key findings

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run preprocessing tests only
pytest tests/test_preprocessing.py -v

# Run API tests only
pytest tests/test_api.py -v
```

---

## 🐳 Docker Instructions

### Using Docker Compose (Recommended)

```bash
# Build and start both API and dashboard
docker-compose up --build

# Services will be available at:
# - API:       http://localhost:8000
# - Dashboard: http://localhost:8501
```

### Using Docker (API only)

```bash
# Build the image
docker build -t arabic-emotion-api .

# Run the container
docker run -p 8000:8000 arabic-emotion-api
```

### Stopping

```bash
docker-compose down
```

---

## 👥 Team Members

| Name                  | GitHub                                                              | Role                                                  |
|-----------------------|---------------------------------------------------------------------|-------------------------------------------------------|
| **Yahia Achouri**     | [@Yahia995](https://github.com/Yahia995)                           | Data acquisition, pipeline engineering, classical ML   |
| **Mohamed Bouchoucha**| [@mohamed-bouchoucha](https://github.com/mohamed-bouchoucha)       | Data quality, feature engineering, evaluation, application |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Repository:** [github.com/Yahia995/data-mining-project](https://github.com/Yahia995/data-mining-project)
- **Dataset:** [Dz-Emotion on HuggingFace](https://huggingface.co/datasets/Houdna-khilouf/Dz-Emotion)
- **ArSEL Lexicon:** [Husseinjd/ArSEL on GitHub](https://github.com/Husseinjd/ArSEL)
- **Linear Project:** [Arabic Emotion Detection Pipeline](https://linear.app/yahia-workspace/project/arabic-emotion-detection-pipeline-78eea28fafd9)
