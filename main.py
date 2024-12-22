import requests
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from bs4 import BeautifulSoup

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
