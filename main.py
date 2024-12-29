import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
import os
from dotenv import load_dotenv
import datetime

# Load environment variables from a .env file
load_dotenv()

# Download VADER lexicon for sentiment analysis
download('vader_lexicon')

# Get API Key from environment variable
API_KEY = os.getenv("API_KEY")


# Function to check if an article is finance-related
def is_financial_article(article):
    # Expanded list of finance-related keywords
    finance_keywords = [
        'stock', 'investment', 'market', 'financial', 'economy', 'profit', 'loss', 'trading',
        'economy', 'banking', 'asset', 'portfolio', 'dividend', 'stocks', 'bonds', 'revenue', 'debt',
        'equity', 'inflation', 'tax', 'fiscal', 'interest rate', 'shares', 'capital', 'merger', 'acquisition',
        'startup', 'venture capital', 'commodities', 'fund', 'cryptocurrency', 'bitcoin', 'blockchain'
    ]

    # Check if any finance keyword is in the article's title or content
    text = f"{article['headline']} {article['summary']}" if article['summary'] else article['headline']
    return any(keyword in text.lower() for keyword in finance_keywords)


# Function to fetch news articles using Finnhub API, including pagination
def fetch_news(api_key, max_articles=100):
    articles = []
    page = 0
    articles_fetched = 0

    while articles_fetched < max_articles:
        timestamp = int(time.time())  # Use current timestamp to avoid cached results
        url = f'https://finnhub.io/api/v1/news?time={timestamp}&token={api_key}&page={page}'

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad status codes
            data = response.json()

            if not data:
                print("No articles found or an error occurred.")
                break

            # Add fetched articles to the list
            new_articles = [{
                "headline": article["headline"],
                "summary": article.get("summary", ""),
                "url": article["url"],
                "datetime": article["datetime"]
            } for article in data]

            articles.extend(new_articles)
            articles_fetched += len(new_articles)
            page += 1  # Move to the next page if more articles are available

            print(f"Fetched {articles_fetched} articles so far...")

            # If the number of articles fetched is less than the maximum requested, break the loop
            if len(new_articles) == 0:
                break

            # Adding a small delay between requests to prevent hitting rate limits
            time.sleep(1)

        except Exception as e:
            print(f"Error fetching news: {e}")
            break

    return articles


# Function to perform sentiment analysis using VADER
def analyze_financial_sentiment(articles):
    sia = SentimentIntensityAnalyzer()
    sentiment_results = []

    # Define positive and negative financial keywords
    positive_keywords = ['growth', 'surge', 'profit', 'increase', 'gain', 'uptrend', 'rally', 'boom', 'bullish', 'rise']
    negative_keywords = ['loss', 'decline', 'drop', 'plunge', 'fall', 'bearish', 'downtrend', 'crash', 'collapse',
                         'slump']

    for article in articles:
        headline = article['headline']
        summary = article['summary']
        publish_date = article['datetime']
        text_to_analyze = f"{headline} {summary}" if summary else headline

        #Convert publish date from unix timestamp to human readable format
        readable_date = datetime.datetime.fromtimestamp(publish_date)

        # Perform initial sentiment analysis using VADER
        sentiment_score = sia.polarity_scores(text_to_analyze)

        # Count occurrences of positive and negative keywords in the text
        positive_count = sum(text_to_analyze.lower().count(keyword) for keyword in positive_keywords)
        negative_count = sum(text_to_analyze.lower().count(keyword) for keyword in negative_keywords)

        # Adjust the sentiment score based on keyword occurrences
        sentiment_score['compound'] += positive_count * 0.10  # Increased weight for positive keywords
        sentiment_score['compound'] -= negative_count * 0.10  # Increased weight for negative keywords

        # Ensure the compound score stays within the range [-1, 1]
        sentiment_score['compound'] = max(-1, min(1, sentiment_score['compound']))

        # Classify the sentiment based on the adjusted score
        sentiment_category = "Neutral"
        if sentiment_score['compound'] > 0.05:
            sentiment_category = "Positive"
        elif sentiment_score['compound'] < -0.05:
            sentiment_category = "Negative"

        sentiment_results.append({
            'headline': headline,
            'sentiment_score': sentiment_score['compound'],
            'category': sentiment_category,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'date': readable_date
        })

    return sentiment_results


# Function to visualize sentiment data
def visualize_sentiment(sentiment_results):
    if not sentiment_results:
        print("No sentiment data to visualize.")
        return

    df = pd.DataFrame(sentiment_results)

    if df.empty or 'sentiment_score' not in df.columns:
        print("Sentiment data is empty or malformed.")
        return

    # Classify sentiments into positive, negative, and neutral
    df['sentiment_category'] = df['sentiment_score'].apply(
        lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
    )

    sentiment_counts = df['sentiment_category'].value_counts()

    # Plot sentiment category distribution
    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'], edgecolor='black')
    plt.title('Sentiment Category Distribution')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=0)
    plt.show()

    # Display a table of articles with their sentiments
    sorted_df = df.sort_values('date', ascending=False)
    print("Top Articles by Sentiment:")
    print(sorted_df[['date', 'headline', 'sentiment_score']].head(10).to_string(index=False))

    print("\nMost Negative Articles:")
    print(sorted_df[['date', 'headline', 'sentiment_score']].tail(10).to_string(index=False))


# Main function
def main():
    # Fetch financial news articles without asking for category
    articles = fetch_news(API_KEY)
    if not articles:
        print("No articles found.")
        return

    print(f"Fetched {len(articles)} articles.")

    # Perform sentiment analysis
    sentiment_results = analyze_financial_sentiment(articles)

    # Display results and visualize
    visualize_sentiment(sentiment_results)


if __name__ == '__main__':
    main()
