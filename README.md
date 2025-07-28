# 📈 Stock News Sentiment Analyzer & Trading Assistant

Our platform offers instant insights powered by real-time news sentiment analysis and a smart trading assistant that uses cutting-edge natural language processing to evaluate financial headlines and determine whether the market tone is positive, negative, or neutral. Designed for both new and seasoned traders, the platform delivers actionable signals and clear visualizations to help you make informed, confident investment decisions.

---

## 🏠 Home

![Home](https://github.com/user-attachments/assets/c2b6d6d1-1a91-46dc-9199-e08372f23c8a)

---

## 📰 Sentiment Analysis

- Scrapes the latest stock news headlines from **Finviz**
- Analyzes sentiment using a trained **PyTorch LSTM** model
- Displays results in a color-coded table
- Integrates with **YouTube API** to fetch relevant videos

![Sentiment Analysis](https://github.com/user-attachments/assets/e04df12f-a835-4e11-a168-f2473a0aec8c)
![Youtube Videos](https://github.com/user-attachments/assets/5707988d-cdc5-48f9-8cb9-e06f5a798ad3)

---

## 🤖 Trading Assistant

- Combines:
  - News sentiment (AI-based)
  - Technical indicators (SMA7 vs SMA14)
  - Risk tolerance (Low / Medium / High)
- Generates final **Buy / Hold / Sell** signal
- Includes beautifully styled dashboard using custom CSS

![Trading Assistant Tab1](https://github.com/user-attachments/assets/2699e4d7-43a4-409a-8a06-4a2cefa78222)

![Tab 2](https://github.com/user-attachments/assets/156edfcc-5fea-4901-ac4e-582985d43d46)

![Tab 3](https://github.com/user-attachments/assets/bbbd25d5-6831-4829-b1a5-c0d5e18908ca)

![Tab 4](https://github.com/user-attachments/assets/8059d17a-2df8-47cd-a187-7c4a5f398359)

---

## 📉 Stock History

- Fetches last 100 days of price data using **Alpha Vantage API**
- Displays **line chart** of closing prices
- Shows fundamentals like:
  - Current price
  - P/E ratio
  - EPS
  - Market Cap

![Stock History](https://github.com/user-attachments/assets/1f181dbd-49f2-4da0-9924-8990940bf99d)

---

## 🧠 Model Evaluation

- Tests the LSTM model on training data
- Shows:
  - **Accuracy**
  - **Classification Report**
  - **Confusion Matrix** (via heatmap)


---

