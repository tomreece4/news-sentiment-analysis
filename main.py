import os
import time
import re
import datetime
import feedparser
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from nltk import download
from nltk.sentiment import SentimentIntensityAnalyzer

# Optional: transformer-based FinBERT for specialized financial sentiment
try:
    from transformers import pipeline
    finbert_pipeline = pipeline(
        "sentiment-analysis",
        model="yiyanghkust/finbert-tone",
        return_all_scores=True
    )
except ImportError:
    finbert_pipeline = None

# Load environment variables
load_dotenv()

# Download VADER lexicon for sentiment analysis
try:
    download('vader_lexicon')
except:
    pass

# Function to fetch news via RSS feeds
def fetch_rss_news(rss_urls, max_articles=100):
    articles = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            if len(articles) >= max_articles:
                break
            pub_time = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_time = time.mktime(entry.published_parsed)
            articles.append({
                'headline': entry.title,
                'summary': re.sub(r'<[^>]+>', '', getattr(entry, 'summary', '')),
                'url': entry.link,
                'datetime': pub_time
            })
        if len(articles) >= max_articles:
            break
    return articles

# Function to perform sentiment analysis using VADER + optional FinBERT
def analyze_financial_sentiment(articles, use_finbert=True):
    sia = SentimentIntensityAnalyzer()
    sentiment_results = []

    pos_keywords = ['growth', 'surge', 'profit', 'increase', 'gain', 'uptrend', 'rally', 'boom', 'bullish', 'rise']
    neg_keywords = ['loss', 'decline', 'drop', 'plunge', 'fall', 'bearish', 'downtrend', 'crash', 'collapse', 'slump']

    for article in articles:
        text = f"{article['headline']} {article['summary']}"
        cleaned = re.sub(r'http\S+|www\.\S+', '', text).lower()

        # VADER scoring
        vader_scores = sia.polarity_scores(cleaned)

        # Keyword-based adjustment
        pos_count = sum(cleaned.count(k) for k in pos_keywords)
        neg_count = sum(cleaned.count(k) for k in neg_keywords)
        adjusted = max(-1, min(1, vader_scores['compound'] + 0.1 * pos_count - 0.1 * neg_count))

        # FinBERT scoring (optional)
        fin_score = 0.0
        if use_finbert and finbert_pipeline:
            try:
                fin_scores = finbert_pipeline(cleaned)[0]
                pos = next((d['score'] for d in fin_scores if d['label'].lower() == 'positive'), 0)
                neg = next((d['score'] for d in fin_scores if d['label'].lower() == 'negative'), 0)
                fin_score = pos - neg
            except Exception:
                fin_score = 0.0

        # Combine VADER + FinBERT (if available)
        compound = (0.6 * adjusted) + (0.4 * fin_score) if finbert_pipeline else adjusted
        compound = max(-1, min(1, compound))

        # Categorize
        if compound > 0.05:
            category = 'Positive'
        elif compound < -0.05:
            category = 'Negative'
        else:
            category = 'Neutral'

        # Readable date
        readable_date = None
        if article['datetime']:
            readable_date = datetime.datetime.fromtimestamp(article['datetime'])

        sentiment_results.append({
            'date': readable_date,
            'headline': article['headline'],
            'url': article['url'],
            'score': compound,
            'category': category,
            'vader': vader_scores['compound'],
            'finbert': fin_score,
            'pos_count': pos_count,
            'neg_count': neg_count
        })

    return sentiment_results

# Function to visualize sentiment data
def visualize_sentiment(results):
    if not results:
        print("No data to visualize.")
        return
    df = pd.DataFrame(results)
    df['category'] = df['score'].apply(
        lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
    )

    counts = df['category'].value_counts().reindex(['Positive','Neutral','Negative'], fill_value=0)
    plt.figure(figsize=(8,5))
    counts.plot(kind='bar', edgecolor='black')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    # Show top and bottom by score
    print("Most Positive Articles:")
    print(df.nlargest(10,'score')[['date','headline','score','vader','finbert']].to_string(index=False))
    print("\nMost Negative Articles:")
    print(df.nsmallest(10,'score')[['date','headline','score','vader','finbert']].to_string(index=False))

# Main
if __name__ == '__main__':
    RSS_FEEDS = [
        'https://feeds.reuters.com/reuters/businessNews',
        'https://www.investing.com/rss/news.rss',
        'https://feeds.marketwatch.com/marketwatch/topstories/'
    ]
    articles = fetch_rss_news(RSS_FEEDS, max_articles=100)
    print(f"Fetched {len(articles)} articles via RSS.")

    sentiment_data = analyze_financial_sentiment(articles)
    visualize_sentiment(sentiment_data)
