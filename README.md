# Credit Risk Intelligence Platform
> Predicting loan default and financial time-series using XGBoost, Temporal Fusion Transformer, and LSTM — with a full ETL pipeline and interactive explainability dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Progress-f59e0b?style=flat-square)

---

## Results at a glance

| Model | Dataset | AUC-ROC | AUC-PR | MAPE |
|---|---|---|---|---|
| XGBoost (baseline) | Lending Club | 0.XXX | 0.XXX | — |
| FT-Transformer | Lending Club | 0.XXX | 0.XXX | — |
| LSTM | S&P 500 + FRED | — | — | X.X% |
| Temporal Fusion Transformer | S&P 500 + FRED | — | — | X.X% |

> Results will be updated as training runs complete. Metrics computed on held-out test set (2020–2024 data).

---

## Problem statement

Credit default prediction is one of the highest-value applications of machine learning in the financial industry. Manual scoring methods are slow, opaque, and fail to capture complex non-linear relationships between macroeconomic conditions and borrower behavior.

This project builds an end-to-end system that:
1. Ingests and processes multi-source financial data (loan records + macroeconomic indicators)
2. Engineers domain-specific features grounded in credit risk theory
3. Trains and compares tree-based and deep learning models
4. Forecasts key financial time series using sequence models
5. Exposes predictions and explanations via an interactive Streamlit application

---

## Demo

<!-- Replace with actual screenshot or GIF once app is deployed -->
![App demo](reports/figures/app_demo.gif)

**Live app:** [credit-risk-demo.streamlit.app](https://share.streamlit.io) *(deploy link coming soon)*

---

## Architecture

```
Raw data sources
  ├── Lending Club (CSV)         → loan-level features, default labels
  ├── FRED API                   → macro variables (rates, unemployment, CPI)
  └── yfinance API               → S&P 500 OHLCV for time-series module
         │
         ▼
  src/etl.py                     → ingestion, validation, schema enforcement
  src/features.py                → feature engineering (ratios, lags, rolling stats)
         │
         ▼
  notebooks/03_baseline_models   → XGBoost / LightGBM + Optuna tuning
  notebooks/04_deep_learning     → FT-Transformer (tabular) + SHAP
  notebooks/05_time_series       → LSTM + Temporal Fusion Transformer
         │
         ▼
  app/streamlit_app.py           → interactive scoring + SHAP explanations
```

---

## Feature engineering highlights

Standard credit scoring uses raw loan fields. This project goes further by:

- **Debt-to-income ratio bins** — non-linear bucketing that captures default cliffs
- **Payment history lags** — rolling 3/6-month patterns of on-time payments
- **Macro-augmented features** — loan origination date joined with Fed Funds Rate and unemployment at time of issuance
- **Interaction terms** — loan grade × interest rate, loan amount × income
- **SMOTE oversampling** — applied only to training fold to avoid data leakage

---

## Model details

### XGBoost (baseline)
Standard gradient boosting with `scale_pos_weight` for class imbalance. Hyperparameters tuned with Optuna (100 trials, 5-fold CV). Chosen as baseline because it mirrors production models used in consumer lending.

### FT-Transformer (tabular deep learning)
Feature Tokenizer + Transformer architecture (Gorishniy et al., 2021) applied to tabular credit data. Each feature is projected to an embedding before self-attention — captures feature interactions that XGBoost misses without manual engineering.

### LSTM + Temporal Fusion Transformer (time series)
Trained on daily S&P 500 returns augmented with FRED macro variables (Fed Funds Rate, VIX, unemployment). TFT provides interpretable attention weights over time — a key advantage for explaining predictions to stakeholders.

---

## Explainability

All models include SHAP-based explanations:

- **Global importance** — which features drive default predictions across the population
- **Local explanations** — why a specific borrower received their score
- **Dependence plots** — how default probability changes as a feature varies

Available interactively in the Streamlit app and as static exports in `reports/figures/`.

---

## How to run

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-portfolio.git
cd credit-risk-portfolio
pip install -r requirements.txt
```

### 2. Download data

```bash
# Lending Club: download manually from Kaggle and place in data/raw/
# https://www.kaggle.com/datasets/wordsforthewise/lending-club

# FRED and yfinance are fetched automatically via API
make data
```

### 3. Run the full pipeline

```bash
make all          # ETL → features → train → evaluate
make train        # train only
make app          # launch Streamlit locally
make test         # run unit tests
```

### 4. Explore notebooks

Run notebooks in order (`01_eda` → `06_explainability`). Each is self-contained with markdown explanations.

---

## Project structure

```
credit-risk-portfolio/
├── data/
│   ├── raw/              # raw files — not committed to git
│   ├── processed/        # cleaned, feature-engineered
│   └── external/         # FRED, macro variables
├── notebooks/            # analysis and modeling notebooks (01–06)
├── src/                  # modular Python source code
│   ├── etl.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app/                  # Streamlit application
├── models/               # serialized model artifacts (.gitignored)
├── reports/
│   ├── figures/          # exported charts
│   └── model_card.md     # model documentation, biases, limitations
├── tests/
├── config.yaml           # centralized hyperparameters and paths
├── Makefile
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## References

- Gorishniy et al. (2021). *Revisiting Deep Learning Models for Tabular Data*. NeurIPS. [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
- Lim et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. International Journal of Forecasting. [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS.
- Kaggle — [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Federal Reserve Bank of St. Louis — [FRED Economic Data](https://fred.stlouisfed.org/)

---

## About

Built by **Jesús Fernando Gómez Brito** — Data Scientist with a background in predictive modeling, ETL engineering, and deep learning for image classification.

- LinkedIn: [linkedin.com/in/jesús-fernando-gómez-brito-02a895279](https://linkedin.com/in/jesús-fernando-gómez-brito-02a895279)
- Email: jesus.gomez1154@alumnos.udg.mx

---

*This project is part of a professional portfolio built to demonstrate end-to-end data science capabilities in the financial domain.*