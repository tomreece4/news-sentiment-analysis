import requests
import matplotlib.pyplot as plt
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download

download('vader_lexicon')  # Download VADER lexicon for sentiment analysis

# Function to fetch news articles using NewsAPI
def fetch_news(query, api_key, max_articles=20):
    url = f'https://newsapi.org/v2/everything?q={query}&pageSize={max_articles}&apiKey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()

        if data.get("status") != "ok" or not data.get("articles"):
            print("No articles found or an error occurred.")
            return []

        articles = [{"title": article["title"], "content": article.get("content", ""), "link": article["url"]} for
                    article in data["articles"] if article["title"]]
        return articles
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


# Function to perform sentiment analysis using VADER
def analyze_financial_sentiment(articles):
    sia = SentimentIntensityAnalyzer()
    sentiment_results = []

    # Define positive and negative financial keywords
    positive_keywords = ['growth', 'surge', 'profit', 'increase', 'gain', 'uptrend', 'rally', 'boom', 'bullish', 'rise']
    negative_keywords = ['loss', 'decline', 'drop', 'plunge', 'fall', 'bearish', 'downtrend', 'crash', 'collapse', 'slump']

    for article in articles:
        title = article['title']
        content = article['content']
        text_to_analyze = f"{title} {content}" if content else title

        # Perform initial sentiment analysis using VADER
        sentiment_score = sia.polarity_scores(text_to_analyze)

        # Count occurrences of positive and negative keywords in the text
        positive_count = sum(text_to_analyze.lower().count(keyword) for keyword in positive_keywords)
        negative_count = sum(text_to_analyze.lower().count(keyword) for keyword in negative_keywords)

        # Adjust the sentiment score based on keyword occurrences
        sentiment_score['compound'] += positive_count * 0.05  # Adjust score positively for each occurrence of a positive keyword
        sentiment_score['compound'] -= negative_count * 0.05  # Adjust score negatively for each occurrence of a negative keyword

        # Ensure the compound score stays within the range [-1, 1]
        sentiment_score['compound'] = max(-1, min(1, sentiment_score['compound']))

        # Classify the sentiment based on the adjusted score
        sentiment_category = "Neutral"
        if sentiment_score['compound'] > 0.05:
            sentiment_category = "Positive"
        elif sentiment_score['compound'] < -0.05:
            sentiment_category = "Negative"

        sentiment_results.append({
            'title': title,
            'sentiment_score': sentiment_score['compound'],  # Store the sentiment score in 'sentiment_score'
            'category': sentiment_category,
            'positive_count': positive_count,
            'negative_count': negative_count
        })

    return sentiment_results


# Function to visualize sentiment data
def visualize_sentiment(sentiment_results):
    if not sentiment_results:
        print("No sentiment data to visualize.")
        return

    df = pd.DataFrame(sentiment_results)

    if df.empty or 'sentiment_score' not in df.columns:  # Fix here, check 'sentiment_score' column
        print("Sentiment data is empty or malformed.")
        return

    # Classify sentiments into positive, negative, and neutral
    df['sentiment_category'] = df['sentiment_score'].apply(  # Update here to use 'sentiment_score'
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
    sorted_df = df.sort_values('sentiment_score', ascending=False)  # Update here to use 'sentiment_score'
    print("Top Articles by Sentiment:")
    print(sorted_df[['title', 'sentiment_score']].head(10).to_string(index=False))

    print("\nMost Negative Articles:")
    print(sorted_df[['title', 'sentiment_score']].tail(10).to_string(index=False))


# Main function
def main():
    api_key = input("Enter your NewsAPI key: ")
    query = input("Enter a topic to search news: ")

    # Fetch news articles
    articles = fetch_news(query, api_key)
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
