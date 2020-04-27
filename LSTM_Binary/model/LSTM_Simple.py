import torch
from torch import nn


class LSTM(nn.Module):
    """
    """
    def __init__(self, input_size, embedding_dim, out_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=128, num_layers=1,
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, out_dim)
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x_emd = self.embedding(x)
        out, (hidden, cell) = self.lstm(x_emd)
        hidden = self.dropout(hidden)
        pred = self.fc(hidden)
        return pred[0]
