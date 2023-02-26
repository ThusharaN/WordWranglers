import torch

class BiLSTMClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embeddings, freeze_embeddings=True):
        super(BiLSTMClassifier, self).__init__()
        self.embedding_layer = torch.nn.Embedding.from_pretrained(embeddings, freeze=freeze_embeddings)
        self.lstm_layer = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_layer = torch.nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, inputs):
        embedded = self.embedding_layer(inputs)
        output, (hidden, _) = self.lstm_layer(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        fc_output = self.fc_layer(hidden)
        return fc_output