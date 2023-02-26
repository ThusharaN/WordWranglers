from torch.utils.data import Dataset

class QuestionDataset(Dataset):
    def __init__(self, questions, coarse_labels, fine_labels, word2idx, max_seq_len):
        self.questions = questions
        self.coarse_labels = coarse_labels
        self.fine_labels = fine_labels
        self.word2idx = word2idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        coarse_label = self.coarse_labels[idx]
        fine_label = self.fine_labels[idx]
        return question, coarse_label, fine_label