import requests
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd

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

        articles = [{"title": article["title"], "content": article.get("content", ""), "link": article["url"]} for article in data["articles"] if article["title"]]
        return articles
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

# Function to perform sentiment analysis
def analyze_sentiment(articles):
    sentiment_results = []
    for article in articles:
        title = article['title']
        content = article['content']

        # Combine title and content for analysis
        text_to_analyze = f"{title} {content}" if content else title

        if text_to_analyze:  # Ensure text is not empty
            sentiment = TextBlob(text_to_analyze).sentiment.polarity
            sentiment_results.append({'title': title, 'content': content, 'sentiment': sentiment})
        else:
            sentiment_results.append({'title': "No Title", 'content': "", 'sentiment': 0})
    return sentiment_results

# Function to visualize sentiment data
def visualize_sentiment(sentiment_results):
    if not sentiment_results:
        print("No sentiment data to visualize.")
        return

    df = pd.DataFrame(sentiment_results)

    if df.empty or 'sentiment' not in df.columns:
        print("Sentiment data is empty or malformed.")
        return

    # Classify sentiments into positive, negative, and neutral
    df['sentiment_category'] = df['sentiment'].apply(
        lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')
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
    sorted_df = df.sort_values('sentiment', ascending=False)
    print("Top Articles by Sentiment:")
    print(sorted_df[['title', 'sentiment']].head(10).to_string(index=False))

    print("\nMost Negative Articles:")
    print(sorted_df[['title', 'sentiment']].tail(10).to_string(index=False))

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
    sentiment_results = analyze_sentiment(articles)

    # Display results and visualize
    visualize_sentiment(sentiment_results)

if __name__ == '__main__':
    main()
