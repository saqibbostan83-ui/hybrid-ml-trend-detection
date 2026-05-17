# 🌐 Hybrid ML System for Detecting Emerging Digital Trends

A hybrid machine learning system that automatically discovers discussion topics from Reddit comments and detects emerging trends using LDA + Change Point Detection.

## 📌 Project Info
- **Student:** Saqib
- **Course:** Data Science
- **Semester:** End Semester Project

## 🎯 What This Project Does
- Discovers hidden topics from Reddit discussions using LDA Topic Modelling
- Tracks topic popularity over time using Time Series Analysis
- Automatically detects emerging trends using Change Point Detection
- Displays everything in an interactive Streamlit dashboard

## 🛠️ Technologies Used
- **Python 3**
- **Pandas, NumPy** — Data handling
- **Scikit-learn** — LDA Topic Modelling
- **Ruptures** — Change Point Detection
- **Plotly** — Interactive charts
- **Streamlit** — Dashboard
- **NLTK** — Text preprocessing
- **WordCloud** — Topic word clouds

## ⚙️ How It Works
1. **Data Collection** — Reddit comments from r/worldnews (2025-2026)
2. **Preprocessing** — Remove URLs, stopwords, special characters
3. **LDA Topic Modelling** — Discover 10 meaningful topics
4. **Time Series** — Track each topic popularity week by week
5. **Change Point Detection** — Automatically flag emerging trends
6. **Dashboard** — Visualize everything interactively

## 📊 Dataset
- **Source:** r/worldnews Reddit comments
- **Size:** 95,008 comments (10,000 used for dashboard)
- **Period:** January 2025 — February 2026

## 📈 Results
- **10 Topics** discovered
- **61 weeks** of data analysed
- **6 Emerging** trends detected
- **6 Growing** trends detected

## 🚀 Live Dashboard
https://hybrid-ml-trend-detection-opakyewvfccsdxwbhwbgks.streamlit.app <- live dashboard
