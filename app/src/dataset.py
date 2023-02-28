from torch.utils.data import Dataset

class QuestionDataset(Dataset):
    def __init__(self, questions, labels, word2idx, max_seq_len):
        self.questions = questions
        self.labels = labels
        self.word2idx = word2idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        labels = self.labels[idx]
        return question, labels