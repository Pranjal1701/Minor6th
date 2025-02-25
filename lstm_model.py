import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# Load dataset
train_data = pd.read_csv("sent_train.csv")
valid_data = pd.read_csv("sent_valid.csv")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        return encoding.input_ids.squeeze(0), self.labels[idx]

# Prepare datasets
train_texts, train_labels = train_data["text"].tolist(), train_data["label"].tolist()
valid_texts, valid_labels = valid_data["text"].tolist(), valid_data["label"].tolist()

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
valid_dataset = SentimentDataset(valid_texts, valid_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

# Define LSTM Model
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

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMSentiment(input_dim=30522, hidden_dim=256, output_dim=3, num_layers=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), torch.tensor(labels).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "lstm_sentiment_model.pth")
print("Model saved!")
