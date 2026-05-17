import feedparser
import pandas as pd
import pickle
import re
import nltk
import json
import os
from datetime import datetime
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

# ── CLEAN TEXT ──────────────────────────
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# ── FETCH LIVE DATA ──────────────────────
print("Fetching live Reddit data...")
feed = feedparser.parse("https://www.reddit.com/r/worldnews/.rss")

posts = []
for entry in feed.entries:
    text = entry.title + " " + entry.summary
    clean = clean_text(text)
    if clean.strip():
        posts.append({
            "body": entry.title,
            "clean_text": clean,
            "timestamp": entry.published,
            "week": datetime.now().strftime("%Y-%m-%d")
        })

df = pd.DataFrame(posts)
print(f"Fetched: {len(df)} posts")

# ── LOAD MODEL ───────────────────────────
with open('vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)
with open('lda_model.pkl', 'rb') as f:
    lda = pickle.load(f)

labels = pd.read_csv('topic_labels.csv')
topic_names = dict(zip(labels['topic_id'], labels['label']))

# ── PREDICT ──────────────────────────────
cv_matrix = cv.transform(df['clean_text'])
topic_dist = lda.transform(cv_matrix)
df['topic_id'] = topic_dist.argmax(axis=1)
df['topic_label'] = df['topic_id'].map(topic_names)
df['confidence'] = topic_dist.max(axis=1).round(3)

# ── TRENDING TODAY ────────────────────────
trending = df['topic_label'].value_counts().reset_index()
trending.columns = ['topic', 'post_count']
trending['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M")

# ── SAVE RESULTS ──────────────────────────
# Existing results load karo ya naya banao
if os.path.exists('live_predictions.csv'):
    existing = pd.read_csv('live_predictions.csv')
    updated = pd.concat([existing, trending], ignore_index=True)
else:
    updated = trending

updated.to_csv('live_predictions.csv', index=False)
print("Saved live_predictions.csv")

# Summary print karo
print("\n=== LIVE TRENDING TOPICS ===")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
for _, row in trending.iterrows():
    bar = "█" * int(row['post_count'])
    print(f"{row['topic']:<35} {bar} ({row['post_count']})")
