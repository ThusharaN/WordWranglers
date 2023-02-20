from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


training_dataset = "./app/data/train.txt"
testing_dataset = "./app/data/test.txt"

def dataset_length(dataset, type):
    with open(dataset, 'r') as file:
        line_count = 0
        for line in file:
            line_count += 1

    print(f"Number of lines in {type} dataset: {line_count}")

dataset_length(training_dataset, "train")
dataset_length(testing_dataset, "test")

question_list = []
fine_class_labels = []
coarse_class_labels = []

def parse_dataset(filename):
    dataset = open(filename, "r")
    tagged_list = []
    for line in dataset:
        line = line.rstrip().split()
        coarse_class = line[0].split(":")[0]
        fine_class = line[0].split(":")[1]
        question = " ".join(line[1:])

        question_list.append(question)
        coarse_class_labels.append(coarse_class)
        fine_class_labels.append(fine_class)

        tagged_list.append((question, coarse_class, fine_class))

    return tagged_list


training_data_list = parse_dataset(training_dataset)
print(training_data_list[:3])
print(len(training_data_list))

test_data_list = parse_dataset(testing_dataset)
print(test_data_list[:3])
print(len(test_data_list))


stop_words = ["?", "the"]

word_counts = Counter(token for question, coarse_class, fine_class in training_data_list 
                     for token in question.lower().split(" ")
                     if token not in stop_words)

print(word_counts.most_common(5))

vocab_size = 1000
vocab = {
    "<PAD>": 0,  # special token used for padding sequences
    "<UNK>": 1,  # special token used for out-of-vocabulary words
}
for i, (word, count) in enumerate(word_counts.most_common(vocab_size - 2)):
    vocab[word] = i + 2


seq_lengths = [len(text.split()) for text in question_list]

plt.hist(seq_lengths, bins=range(min(seq_lengths), max(seq_lengths) + 2, 1), align='left')
plt.xlabel("Sequence length")
plt.ylabel("Frequency")
plt.show()

max_seq_length = 10   # maximum sequence length to use for padding identified from the plot

coarse_class_encoded_labels = LabelEncoder().fit_transform(coarse_class_labels)    
fine_class_encoded_labels = LabelEncoder().fit_transform(fine_class_labels)  


