# 🌐 Hybrid ML System for Detecting Emerging Digital Trends

A hybrid machine learning system that automatically discovers discussion topics from Reddit comments, tracks their popularity over time, and detects emerging trends — with a live 24/7 data pipeline and interactive dashboard.

## 🔴 Live Dashboard
[Click here to view the live dashboard](https://hybrid-ml-trend-detection-opakyewvfccsdxwbhwbgks.streamlit.app)

---

## 📌 Project Info
- **Course:** Data Science — End Semester Project
- **Date:** May 2026
- **Dataset:** r/worldnews Reddit Comments (2025–2026)

---

## 🎯 What This Project Does
- Automatically discovers 10 meaningful discussion topics from 95,000+ Reddit comments using LDA Topic Modelling — no human labels required
- Tracks each topic's popularity week by week across 61 weeks of data
- Detects emerging trends using Change Point Detection — flags topics whose growth exceeds 2x their baseline
- Fetches live Reddit data from 40+ subreddits every hour via GitHub Actions
- Displays everything in an interactive public Streamlit dashboard

---

## ⚙️ How It Works

**Stage 1 — Data Preprocessing**
Raw Reddit comments cleaned through URL removal, stopword filtering, special character removal, and empty row filtering. 200,000 comments reduced to 170,666 clean records.

**Stage 2 — LDA Topic Modelling**
CountVectorizer (5,000 features) + Latent Dirichlet Allocation discovers 10 topics automatically. Best configuration selected using coherence score.

**Stage 3 — Time Series Construction**
Each comment assigned a dominant topic. Weekly aggregation creates a 61-week × 10-topic time series matrix.

**Stage 4 — Change Point Detection**
ruptures library (PELT algorithm) detects exact weeks where topic popularity shifts. Topics with >2x growth flagged as EMERGING.

**Stage 5 — Live Data Pipeline**
GitHub Actions fetches fresh posts from 40+ subreddits every hour, classifies them using the trained LDA model, and updates live_predictions.csv automatically — 24/7, no manual intervention.

**Stage 6 — Dashboard**
Interactive Streamlit app with 4 pages: Overview, Topic Explorer, Emerging Trends, and Live Data.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Original dataset | 1,000,000+ comments |
| After preprocessing | 170,666 clean comments |
| Topics discovered | 10 |
| Weeks analysed | 61 |
| Emerging trends detected | 6 |
| Growing trends detected | 6 |
| Live subreddits monitored | 40+ |
| Peak week | 3,133 comments (March 3, 2025) |

---

## 💡 Topics Discovered

| ID | Topic |
|----|-------|
| 0 | Military & Defense |
| 1 | Politics & Government |
| 2 | International Affairs |
| 3 | Social Media & Public Opinion |
| 4 | Nuclear & Security Affairs |
| 5 | Global Issues |
| 6 | Regional Conflicts |
| 7 | Security & Terrorism |
| 8 | Energy & Commodities |
| 9 | Diplomacy & Relations |

---

## 🚀 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3 |
| Data | Pandas, NumPy |
| NLP | NLTK, CountVectorizer |
| Topic Modelling | scikit-learn LDA |
| Trend Detection | ruptures (PELT/BinSeg) |
| Visualisation | Plotly, WordCloud, Matplotlib |
| Dashboard | Streamlit |
| Live Data | feedparser (RSS), GitHub Actions |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |
| Environment | Google Colab |

---

## 🔴 Live Data Pipeline

The system automatically fetches live Reddit posts from 40+ subreddits every hour using GitHub Actions — completely free, no server required. Each batch is classified using the trained LDA model and results are saved to `live_predictions.csv`. The dashboard reads this file every minute and updates automatically.

**Subreddits monitored:** worldnews, news, geopolitics, economy, politics, technology, science, business, finance, energy, climate, war, ukraine, russia, china, india, europe, middleeast, africa, asia, and 20+ more.

---

## 📂 Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit dashboard |
| `fetch_and_predict.py` | Live data fetch + prediction script |
| `.github/workflows/live_fetch.yml` | GitHub Actions scheduler |
| `clean_reddit_comments.csv` | Cleaned Reddit dataset |
| `topic_time_series.csv` | Weekly topic counts |
| `topic_labels.csv` | Topic labels and keywords |
| `emerging_trends_results.csv` | Change point detection results |
| `live_predictions.csv` | Live predictions (auto-updated) |
| `lda_model.pkl` | Trained LDA model |
| `vectorizer.pkl` | Trained CountVectorizer |

---

## ⚡ Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

*This project is academically strong, computationally efficient, and uses zero Large Language Models — pure classical ML.*
