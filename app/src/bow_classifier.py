import torch

class Classifier(torch.nn.Module):
    def __init__(self, num_words, embedding_dim, num_coarse_classes, num_fine_classes):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_words, embedding_dim)
        self.fc1 = torch.nn.Linear(embedding_dim, 128)
        self.fc2 = torch.nn.Linear(128, num_coarse_classes)
        self.fc3 = torch.nn.Linear(128, num_fine_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim = 1)
        x = torch.relu(self.fc1(x))
        coarse_output = self.fc2(x)
        fine_output = self.fc3(x)
        coarse_output = self.softmax(coarse_output)
        fine_output = self.softmax(fine_output)
        return coarse_output, fine_output
