import requests
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from bs4 import BeautifulSoup

# Function to fetch news articles by web scraping
def fetch_news(query, max_articles=10):
    url = f'https://news.google.com/search?q={query}'
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching news: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    headlines = soup.find_all('a', class_='DY5T1d', limit=max_articles)

    articles = []
    for headline in headlines:
        title = headline.get_text()
        link = 'https://news.google.com' + headline['href'][1:]
        articles.append({'title': title, 'link': link})

    return articles

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
    query = input("Enter a topic to search news: ")

    # Fetch news articles
    articles = fetch_news(query)
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
