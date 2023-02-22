import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import utils
import yaml
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


def split_training_dataset(
    training_dataset, split_fraction, training_datapath, validation_datapath
):
    # Split the given data set into training and vaidation sets with the given split condition
    with open(training_dataset, "r") as f:
        contents = f.readlines()
    split_idx = int(len(contents) * split_fraction)
    random.shuffle(contents)
    dev_data = contents[:split_idx]
    train_data = contents[split_idx:]
    with open(validation_datapath, "w") as f:
        f.writelines(dev_data)
    with open(training_datapath, "w") as f:
        f.writelines(train_data)


def train(config_file):
    torch.manual_seed(1)
    random.seed(1)

    with open(
        os.getcwd().replace("src", "data/") + config_file, "r"
    ) as configurations:
        config = yaml.load(configurations, Loader=yaml.Loader)

    utils.dataset_length(config["original_dataset"], "train")
    split_training_dataset(
        config["original_dataset"],
        config["split_ratio"],
        config["training_dataset"],
        config["validation_dataset"],
    )
    (
        training_data_list,
        question_list,
        coarse_class_labels,
        fine_class_labels,
    ) = utils.parse_dataset(config["original_dataset"])
    word_counts = Counter(
        token
        for question, coarse_class, fine_class in training_data_list
        for token in question.lower().split(" ")
        if token not in utils.STOP_WORDS
    )
    vocab = {
        "<PAD>": 0,  # special token used for padding sequences
        "<UNK>": 1,  # special token used for out-of-vocabulary words
    }
    for i, (word, count) in enumerate(
        word_counts.most_common(config["vocab_size"] - 2)
    ):
        if count >= config["min_freq"]:
            vocab[word] = i + 2
    print(f"Vocabulary length {len(vocab)}")

    # Randomly initialised word embeddings
    embedding = nn.Embedding(config["vocab_size"], config["word_embedding_dim"])
    max_seq_length = config["max_seq_length"]
    inputs = []
    for text in question_list:
        tokens = text.split()
        indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        if len(indices) < max_seq_length:
            indices += [vocab["<PAD>"]] * (max_seq_length - len(indices))
        indices = indices[:max_seq_length]
        tensor = torch.tensor(indices)
        inputs.append(tensor)
    inputs = torch.stack(inputs)
    embedded = embedding(inputs)
    print(embedded.shape)

    # GloVe
    # glove_path = './app/data/glove.6B.50d.txt'
    # vocab, embeddings = [], []
    # with open(glove_path, 'rt', encoding='utf-8') as fi:
    #     full_content = fi.read().strip().split('\n')
    # for i in range(len(full_content)):
    #     i_word = full_content[i].split(' ')[0]
    #     i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    #     vocab.append(i_word)
    #     embeddings.append(i_embeddings)

    # vocab_npa = np.array(vocab)
    # embs_npa = np.array(embeddings)

    # # insert '<pad>' and '<unk>' tokens at start of vocab_npa.
    # vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    # vocab_npa = np.insert(vocab_npa, 1, '<unk>')
    # print(vocab_npa[:10])

    # # embedding for '<pad>' token.
    # pad_emb_npa = np.zeros((1, embs_npa.shape[1]))
    # # embedding for '<unk>' token.
    # unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True)

    # # insert embeddings for pad and unk tokens at top of embs_npa.
    # embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, embs_npa))
    # print(embs_npa.shape)

    # my_embedding_layer = torch.nn.Embedding.from_pretrained(
    #     torch.from_numpy(embs_npa).float())

    # assert my_embedding_layer.weight.shape == embs_npa.shape
    # print(my_embedding_layer.weight.shape)

    # TODO: Validation data
    # test_data_list = parse_dataset(
    #     constants.TESTING_DATA, question_list, fine_class_labels, coarse_class_labels, max_sentence_length)
    # dataset_length(constants.TESTING_DATA, "test")
