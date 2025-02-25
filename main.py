import streamlit as st
import requests
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer

# Load model
class LSTMSentiment(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMSentiment, self).__init__()
        self.embedding = nn.Embedding(input_dim, 128)
        self.lstm = nn.LSTM(128, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(self.dropout(lstm_out[:, -1, :]))
        return output

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMSentiment(input_dim=30522, hidden_dim=256, output_dim=3, num_layers=2).to(device)
model.load_state_dict(torch.load("lstm_sentiment_model.pth", map_location=device))
model.eval()

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def predict_sentiment(text):
    tokens = tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    input_ids = tokens.input_ids.to(device)
    with torch.no_grad():
        output = model(input_ids)
    sentiment = torch.argmax(output, dim=1).item()
    return "Positive" if sentiment == 2 else "Neutral" if sentiment == 1 else "Negative"

# Function to fetch real-time news from Finviz
def get_finviz_news(ticker):
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    news_table = soup.find(id='news-table')
    
    parsed_news = []
    if news_table:
        for row in news_table.findAll('tr'):
            title = row.a.text if row.a else "No Title"
            parsed_news.append(title)
    
    return parsed_news

# Streamlit UI
st.title("Real-Time Stock News Sentiment Analyzer")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")

if st.button("Fetch News & Analyze"):
    news_headlines = get_finviz_news(ticker.upper())
    
    if news_headlines:
        st.subheader(f"Latest News for {ticker.upper()}")
        results = []
        for news in news_headlines[:50]:  # Analyze top 10 news
            sentiment = predict_sentiment(news)
            results.append([news, sentiment])
        
        df = pd.DataFrame(results, columns=["News Headline", "Predicted Sentiment"])
        st.dataframe(df)
    else:
        st.error("Failed to fetch news. Check the ticker or try again later.")
