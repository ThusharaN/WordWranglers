from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import string
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
torch.manual_seed(1)
random.seed(1)

training_dataset = "./app/data/train_5500_label.txt"
testing_dataset = "./app/data/test.txt"

def dataset_length(dataset, type):
    with open(dataset, 'r') as file:
        line_count = 0
        for line in file:
            line_count += 1

    print(f"Number of lines in {type} dataset: {line_count}")

dataset_length(training_dataset, "train")
dataset_length(testing_dataset, "test")


def split_training_dataset(split_fraction):
    global training_dataset

    with open(training_dataset, 'r') as f:
        contents = f.readlines()
    
    split_idx = int(len(contents) * split_fraction)
    
    random.shuffle(contents)

    dev_data = contents[:split_idx]
    train_data = contents[split_idx:]
    
    training_dataset = './app/data/training.txt'

    with open('./app/data/dev.txt', 'w') as f:
        f.writelines(dev_data)

    with open(training_dataset, 'w') as f:
        f.writelines(train_data)

split_training_dataset(0.1)         #10:90

question_list = []
fine_class_labels = []
coarse_class_labels = []
max_sentence_length = 0 

def parse_dataset(filename):
    global max_sentence_length

    dataset = open(filename, "r")
    tagged_list = []
    for line in dataset:
        line = line.rstrip().split()
        if(max_sentence_length < (len(line) - 1)):
            max_sentence_length = (len(line) - 1)

        coarse_class = line[0].split(":")[0]
        fine_class = line[0].split(":")[1]
        question = " ".join(line[1:])

        question_list.append(question)
        coarse_class_labels.append(coarse_class)
        fine_class_labels.append(fine_class)

        tagged_list.append((question, coarse_class, fine_class))

    return tagged_list

training_data_list = parse_dataset(training_dataset)
#print(training_data_list[:3])
#print(len(training_data_list))

test_data_list = parse_dataset(testing_dataset)
#print(test_data_list[:3])
#print(len(test_data_list))

print(f"Maximum sentence: {max_sentence_length}")

stop_words = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'] + [punct for punct in string.punctuation]

word_counts = Counter(token for question, coarse_class, fine_class in training_data_list 
                     for token in question.lower().split(" ")
                     if token not in stop_words)

#print(word_counts.most_common(5))

vocab_size = 10000
min_freq = 5 #hyperparameter 
vocab = {
    "<PAD>": 0,  # special token used for padding sequences
    "<UNK>": 1,  # special token used for out-of-vocabulary words
}

for i, (word, count) in enumerate(word_counts.most_common(vocab_size - 2)):
    if count >= min_freq:
        vocab[word] = i + 2

print(f'Vocabulary length {len(vocab)}')

# seq_lengths = [len(text.split()) for text in question_list]

# plt.hist(seq_lengths, bins=range(min(seq_lengths), max(seq_lengths) + 2, 1), align='left')
# plt.xlabel("Sequence length")
# plt.ylabel("Frequency")
# plt.show()

max_seq_length = 10   # maximum sequence length to use for padding identified from the plot

coarse_class_encoded_labels = LabelEncoder().fit_transform(coarse_class_labels)    
fine_class_encoded_labels = LabelEncoder().fit_transform(fine_class_labels)  

#print(set(coarse_class_encoded_labels))
#print(set(fine_class_encoded_labels))

word_embedding_dim = 200
batch_size = 20

embedding = nn.Embedding(vocab_size, word_embedding_dim)

inputs = []
for text in question_list:
    # Tokenize the text
    tokens = text.split()
    # Map the tokens to their corresponding indices in the vocabulary
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    # Pad the sequence if necessary
    if len(indices) < max_seq_length:
        indices += [vocab["<PAD>"]] * (max_seq_length - len(indices))
    # Truncate the sequence if necessary
    indices = indices[:max_seq_length]
    # Convert the indices to a tensor
    tensor = torch.tensor(indices)
    inputs.append(tensor)

# Stack the input tensors into a matrix
inputs = torch.stack(inputs)
embedded = embedding(inputs)

print(embedded.shape)

