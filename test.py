import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

def get_general_stock_news():
    """Fetch the latest general stock market news from MarketWatch."""
    url = 'https://www.marketwatch.com/latest-news?mod=top_nav'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    news_items = soup.find_all('div', class_='article__content')
    
    news_data = []
    for item in news_items[:15]:  # Get top 15 news articles
        title_tag = item.find('h3', class_='article__headline')
        time_tag = item.find('span', class_='article__timestamp')
        link_tag = item.find('a', href=True)
        
        if title_tag and link_tag:
            title = title_tag.text.strip()
            link = link_tag['href']
            timestamp = time_tag.text.strip() if time_tag else datetime.now().strftime("%Y-%m-%d %H:%M")
            news_data.append([timestamp, title, link])
    
    return news_data

st.title("General Stock Market News Scraper")

if st.button("Fetch General News"):
    news_data = get_general_stock_news()
    
    if news_data:
        df = pd.DataFrame(news_data, columns=['Date & Time', 'Title', 'Link'])
        st.subheader("Latest Stock Market News")
        st.dataframe(df)
    else:
        st.error("Failed to fetch news. Try again later.")
