import torch.nn as nn
import torch
import random
import numpy as np
import torch.nn.functional as F


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)  
    torch.cuda.manual_seed_all(1234)
    

class EnsembleClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embeddings,freeze, padding_token):
        super(EnsembleClassifier, self).__init__()
        self.embedded = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings), freeze=freeze, padding_idx=padding_token)
        self.lstm_layer = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm_hidden = nn.Linear(hidden_size * 2, output_size)
        self.bow_hidden = torch.nn.Linear(300, output_size)

    def forward(self, x):
        sent_vec= []
        embedded = self.embedded(x)
        _, (hidden, _) = self.lstm_layer(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        lstm_1 = self.lstm_hidden(hidden)
        lstm_scores = F.log_softmax(lstm_1.view(1, -1), dim=1)
        for n in range(0, len(embedded)):
            sent_vec.append(torch.mean(embedded[n], 0))
        sent_vec_tensor = torch.stack(sent_vec)
        bow_1 = self.bow_hidden(sent_vec_tensor)
        bow_scores = F.log_softmax(bow_1.view(1, -1), dim=1)
        final_scores = torch.mean(torch.stack([bow_scores, lstm_scores]), 0)
        return final_scores
    

