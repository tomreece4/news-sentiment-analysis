import requests
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


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

        articles = [{"title": article["title"], "link": article["url"]} for article in data["articles"] if
                    article["title"]]
        return articles
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


# Function to perform sentiment analysis
def analyze_sentiment(articles):
    sentiment_results = []
    for article in articles:
        title = article['title']
        if title:  # Ensure title is not empty
            sentiment = TextBlob(title).sentiment.polarity
            sentiment_results.append({'title': title, 'sentiment': sentiment})
        else:
            sentiment_results.append({'title': "No Title", 'sentiment': 0})
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

    # Plot sentiment distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['sentiment'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.show()

    # Generate a WordCloud of positive titles
    positive_titles = ' '.join(df[df['sentiment'] > 0]['title'])
    if positive_titles:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_titles)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('WordCloud of Positive News Titles')
        plt.show()
    else:
        print("No positive titles to generate a WordCloud.")


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


