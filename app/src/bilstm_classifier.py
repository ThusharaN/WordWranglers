import torch
import torch.nn as nn
import random
import numpy as np


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)  
    torch.cuda.manual_seed_all(1234)

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embeddings, freeze_embeddings):
        super(BiLSTMClassifier, self).__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=freeze_embeddings)
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_layer = nn.Linear(hidden_dim * 2, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        embedded = self.embedding_layer(inputs)
        _, (hidden, _) = self.lstm_layer(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.relu(hidden)
        fc_output = self.fc_layer(hidden)
        return fc_output