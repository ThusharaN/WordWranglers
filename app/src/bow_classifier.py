import torch.nn as nn
import torch

class BoWClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embeddings, padding_token):
        super(BoWClassifier, self).__init__()
        self.embedded = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings), freeze=True, padding_idx=padding_token)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        embedded = self.embedded(x)
        sent_vec= []
        for n in range(0, len(embedded)):
            sent_vec.append(torch.mean(embedded[n], 0))
        sent_vec_tensor = torch.stack(sent_vec)
        out = self.fc1(sent_vec_tensor)
        out = self.relu(out)
        out = self.fc2(out)
        return out
