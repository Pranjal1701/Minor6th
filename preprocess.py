import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Tokenizer for text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")

def preprocess_text(text):
    """Cleans and preprocesses text data"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

def load_dataset(file_path):
    """Loads and preprocesses financial sentiment dataset"""
    df = pd.read_csv(file_path)
    df["processed_text"] = df["text"].apply(preprocess_text)
    return df

def tokenize_text(texts):
    """Tokenizes and converts texts to padded sequences"""
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=20)
