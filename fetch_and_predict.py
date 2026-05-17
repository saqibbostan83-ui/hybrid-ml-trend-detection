import feedparser
import pandas as pd
import pickle
import re
import nltk
import os
from datetime import datetime
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# ── 40 SUBREDDITS ─────────────────────────
subreddits = [
    "worldnews", "news", "geopolitics", "economy", "environment",
    "politics", "technology", "science", "business", "finance",
    "energy", "climate", "war", "ukraine", "russia",
    "china", "india", "europe", "middleeast", "africa",
    "asia", "australia", "canada", "unitedkingdom", "germany",
    "france", "japan", "korea", "pakistan", "turkey",
    "israel", "iran", "space", "health", "covid19",
    "military", "diplomacy", "humanrights", "refugees", "economics"
]

# ── FETCH ALL SUBREDDITS ──────────────────
print(f"Fetching from {len(subreddits)} subreddits...")
all_posts = []

for sub in subreddits:
    try:
        feed = feedparser.parse(f"https://www.reddit.com/r/{sub}/.rss")
        for entry in feed.entries:
            text = entry.title + " " + entry.get("summary", "")
            clean = clean_text(text)
            if clean.strip():
                all_posts.append({
                    "body": entry.title,
                    "clean_text": clean,
                    "subreddit": sub,
                    "timestamp": entry.published,
                    "week": datetime.now().strftime("%Y-%m-%d")
                })
        print(f"  ✅ r/{sub}: {len(feed.entries)} posts")
    except Exception as e:
        print(f"  ❌ r/{sub}: {e}")

df = pd.DataFrame(all_posts)
print(f"\nTotal posts fetched: {len(df)}")

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

# ── TRENDING ─────────────────────────────
trending = df['topic_label'].value_counts().reset_index()
trending.columns = ['topic', 'post_count']
trending['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M")

# ── SAVE ─────────────────────────────────
if os.path.exists('live_predictions.csv'):
    existing = pd.read_csv('live_predictions.csv')
    updated = pd.concat([existing, trending], ignore_index=True)
else:
    updated = trending

updated = updated.sort_values('timestamp', ascending=False)
updated.to_csv('live_predictions.csv', index=False)

print("\n=== LIVE TRENDING TOPICS ===")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
for _, row in trending.iterrows():
    bar = "█" * int(row['post_count'])
    print(f"{row['topic']:<35} {bar} ({row['post_count']})")

print(f"\nTotal: {len(df)} posts from {len(subreddits)} subreddits")
