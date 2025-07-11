import datetime
import time
import feedparser
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from nltk import download
from nltk.sentiment import SentimentIntensityAnalyzer

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
                'summary': getattr(entry, 'summary', ''),
                'url': entry.link,
                'datetime': pub_time
            })
        if len(articles) >= max_articles:
            break
    return articles

# Function to perform sentiment analysis using VADER
def analyze_financial_sentiment(articles):
    sia = SentimentIntensityAnalyzer()
    sentiment_results = []

    positive_keywords = ['growth', 'surge', 'profit', 'increase', 'gain', 'uptrend', 'rally', 'boom', 'bullish', 'rise']
    negative_keywords = ['loss', 'decline', 'drop', 'plunge', 'fall', 'bearish', 'downtrend', 'crash', 'collapse', 'slump']

    for article in articles:
        text = f"{article['headline']} {article['summary']}".lower()
        score = sia.polarity_scores(text)

        # Adjust weights
        pos_count = sum(text.count(k) for k in positive_keywords)
        neg_count = sum(text.count(k) for k in negative_keywords)
        score['compound'] = max(-1, min(1, score['compound'] + 0.1 * pos_count - 0.1 * neg_count))

        category = 'Neutral'
        if score['compound'] > 0.05:
            category = 'Positive'
        elif score['compound'] < -0.05:
            category = 'Negative'

        readable_date = None
        if article['datetime']:
            readable_date = datetime.datetime.fromtimestamp(article['datetime'])

        sentiment_results.append({
            'date': readable_date,
            'headline': article['headline'],
            'url': article['url'],
            'score': score['compound'],
            'category': category,
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
    df['category'] = df['score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

    counts = df['category'].value_counts()
    plt.figure(figsize=(8,5))
    counts.plot(kind='bar', edgecolor='black')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    print(df.sort_values('date', ascending=False)[['date','headline','score']].head(10).to_string(index=False))

# Main
if __name__ == '__main__':
    # Define RSS feeds for financial news
    RSS_FEEDS = [
        'https://feeds.reuters.com/reuters/businessNews',
        'https://www.investing.com/rss/news.rss',
        'https://feeds.marketwatch.com/marketwatch/topstories/'
    ]
    articles = fetch_rss_news(RSS_FEEDS, max_articles=100)
    print(f"Fetched {len(articles)} articles via RSS.")

    sentiment_data = analyze_financial_sentiment(articles)
    visualize_sentiment(sentiment_data)

