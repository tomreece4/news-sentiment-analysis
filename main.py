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
