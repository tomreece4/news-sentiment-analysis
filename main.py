print("Importing Packages...")
import time
import re
import datetime
import argparse
import feedparser
import pandas as pd
from nltk import download
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm


download('vader_lexicon')

# Function to fetch news via RSS feeds
def fetch_rss_news(rss_urls, max_articles=100):
    articles = []
    seen_urls = set()

    for url in tqdm(rss_urls, desc="Fetching RSS Feeds..."):
        feed = feedparser.parse(url)
        for entry in feed.entries:
            if len(articles) >= max_articles:
                break

            if entry.link in seen_urls:
                continue
            seen_urls.add(entry.link)

            pub_time = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_time = time.mktime(entry.published_parsed)

            articles.append({
                'headline': getattr(entry, 'title', 'No Title'),
                'summary': re.sub(r'<[^>]+>', '', getattr(entry, 'summary', '')),
                'url': entry.link,
                'datetime': pub_time
            })

        if len(articles) >= max_articles:
            break

    return articles

# Function to perform sentiment analysis using VADER + optional FinBERT
def analyze_financial_sentiment(articles, use_finbert=True):

    if use_finbert:
        print("Loading finbert model...")
        from transformers import pipeline
        finbert_pipeline = pipeline(
            task="sentiment-analysis",
            model="yiyanghkust/finbert-tone",
            return_all_scores=True
        )
        print("Finbert model loaded")

    sia = SentimentIntensityAnalyzer()
    sentiment_results = []

    pos_keywords = ['growth', 'surge', 'profit', 'increase', 'gain', 'uptrend', 'rally', 'boom', 'bullish', 'rise']
    neg_keywords = ['loss', 'decline', 'drop', 'plunge', 'fall', 'bearish', 'downtrend', 'crash', 'collapse', 'slump']

    for article in tqdm(articles, desc="Analyzing Articles...", unit="article"):
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
        compound = (0.6 * adjusted) + (0.4 * fin_score) if use_finbert and finbert_pipeline else adjusted
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
            try:
                readable_date = datetime.datetime.fromtimestamp(article['datetime'])
            except Exception:
                readable_date = None

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

# Debug-safe and table-based visualization
def visualize_sentiment(results):
    if not results:
        print("No data to visualize.")
        return

    df = pd.DataFrame(results)

    # Ensure datetime is parsed safely
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Handle missing expected columns
    required_cols = ['score', 'headline', 'date', 'vader', 'finbert', 'url']
    for col in required_cols:
        if col not in df.columns:
            print(f"Missing column: {col}")
            return

    # Classify category if not present
    if 'category' not in df.columns:
        df['category'] = df['score'].apply(
            lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
        )

    # Optional: truncate long headlines for display
    df['headline'] = df['headline'].str.slice(0, 100) + '...'

    top_positive = df.nlargest(10, 'score')[['date', 'headline', 'score', 'vader', 'finbert', 'url']]
    top_negative = df.nsmallest(10, 'score')[['date', 'headline', 'score', 'vader', 'finbert', 'url']]

    print("\nTop 10 Most Positive Articles:\n")
    for _, row in top_positive.iterrows():
        print(f"{row['date']} | {row['headline']} | Score: {row['score']:.3f} | VADER: {row['vader']:.3f} | FinBERT: {row['finbert']:.3f}")
        print(f"Link: {row['url']}\n")

    print("\nTop 10 Most Negative Articles:\n")
    for _, row in top_negative.iterrows():
        print(f"{row['date']} | {row['headline']} | Score: {row['score']:.3f} | VADER: {row['vader']:.3f} | FinBERT: {row['finbert']:.3f}")
        print(f"Link: {row['url']}\n")


# ------------------------
# Main script with CLI arg
# ------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Financial News Sentiment Analyzer")
    parser.add_argument(
        "--no-finbert",
        action="store_true",
        help="Disable FinBERT for sentiment analysis"
    )
    args = parser.parse_args()

    use_finbert = not args.no_finbert

    RSS_FEEDS = [
        'https://feeds.reuters.com/reuters/businessNews',
        'https://www.investing.com/rss/news.rss',
        'https://feeds.marketwatch.com/marketwatch/topstories/'
    ]

    articles = fetch_rss_news(RSS_FEEDS, max_articles=100)
    print(f"Fetched {len(articles)} articles via RSS.")

    sentiment_data = analyze_financial_sentiment(articles, use_finbert=use_finbert)
    visualize_sentiment(sentiment_data)
