import random
import torch.nn as nn
import torch
import numpy as np


torch.manual_seed(1)
torch.cuda.manual_seed(1)                                                                                                                              
torch.cuda.manual_seed_all(1)                                                                                          
np.random.seed(1)                                                                                                             
random.seed(1) 


STOP_WORDS = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
]

PUNC = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"

def dataset_length(dataset, type):
    # Verifying the given dataset
    with open(dataset, "r", encoding="ISO-8859-1") as file:
        line_count = 0
        for line in file:
            line_count += 1
    print(f"Number of lines in {type} dataset: {line_count}")


def parse_dataset(filename, prediction_class):
    # Parsing the given dataset to return list of questions, coarse and fine classes
    dataset = open(filename, "r", encoding="ISO-8859-1")
    parsed_list = []
    question_list = []
    fine_class_labels = []
    coarse_class_labels = []
    max_sentence_length = 0
    for line in dataset:
        line = line.rstrip().split()
        if max_sentence_length < (len(line) - 1):
            max_sentence_length = len(line) - 1
        coarse_class = line[0].split(":")[0]
        fine_class = line[0].split(":")[1]
        question = " ".join(line[1:])
        question_list.append(question)
        coarse_class_labels.append(coarse_class)
        fine_class_labels.append(fine_class)
        if(prediction_class == "fine"):
            parsed_list.append((question, fine_class))
        else:
            parsed_list.append((question, coarse_class))
    print(f"Longest sentence in file {filename} is of length: {max_sentence_length}")
    return question_list, coarse_class_labels, fine_class_labels, max_sentence_length, parsed_list


def get_randomly_initialised_bow(question_list, min_freq, word_embedding_dim, max_seq_length, remove_stopwords):
    # Randomly initialised word embeddings
    print("Starting bag of words...")
    custom_vocab_index = {
        "#PAD#": 0,  # special token used for padding sequences
        "#UNK#": 1,  # special token used for out-of-vocabulary words
    }
    # Removing the words that are not frequent
    custom_vocab_list = []
    for ques in question_list:
        custom_vocab_list = custom_vocab_list + ques.lower().split(' ')
    if(remove_stopwords):
        custom_vocab_list = [
            i for i in custom_vocab_list if i not in STOP_WORDS]
    custom_vocab_count = {}
    for i in custom_vocab_list:
        if i not in list(custom_vocab_count.keys()):
            custom_vocab_count[i] = 1
        else:
            custom_vocab_count[i] = custom_vocab_count[i] + 1
    custom_vocab_count_frequency = {
        k: v for k, v in custom_vocab_count.items() if v >= min_freq}
    word_indices = 2
    for w in custom_vocab_count_frequency.keys():
        custom_vocab_index[w] = word_indices
        word_indices = word_indices + 1

    # Get word embeddings based on the indices created earlier
    random_init_embedding = nn.Embedding(
        len(custom_vocab_count_frequency)+2, word_embedding_dim)
    sent_idx_random = []
    for question in question_list:
        tokens = question.lower().split()
        sent_indices = [custom_vocab_index.get(
            token, custom_vocab_index["#UNK#"]) for token in tokens]
        if len(sent_indices) < max_seq_length:
            sent_indices += [custom_vocab_index["#PAD#"]] * \
                (max_seq_length - len(sent_indices))
        sent_indices = sent_indices[:max_seq_length]
        sent_indices_tensor = torch.tensor(sent_indices)
        sent_idx_random.append(sent_indices_tensor)
    sent_idx_random = torch.stack(sent_idx_random)
    random_init_embedded = random_init_embedding(sent_idx_random)

    # Create bag-of-words from randomly initialized embedding
    bow_random = {}
    for q in range(0, len(question_list)):
        #tokens = question_list[q].lower().split()
        sent_vec = random_init_embedded[q].sum(0)
        #sent_vec = torch.div(sent_vec, len(tokens))
        sent_vec = torch.div(sent_vec, random_init_embedded)
        bow_random[question_list[q]] = sent_vec
    return bow_random


def get_pre_trained_bow(question_list, pretrained_glove, remove_stopwords):
    # Pre-trained word embeddings using GloVe
    # glove_vocab,glove_embeddings = [],[]
    print("Starting GloVe for the given dataset...")
    emb_dict_glove_bow = {}
    bow_glove = {}
    with open(pretrained_glove, 'rt', encoding="ISO-8859-1") as pretrained_file:
        pretrained_vectors = pretrained_file.read().strip().split('\n')  # read().strip().split('\t')
    for i in range(len(pretrained_vectors)):
        word = pretrained_vectors[i].split('\t')[0]
        word_vector = [float(val)
                       for val in pretrained_vectors[i].split('\t')[1].split(' ')[0:]]
        emb_dict_glove_bow[word] = word_vector
    for q in question_list:
        tokens = q.lower().split()
        # REMOVE STOP WORDS HERE
        word_vecs = [emb_dict_glove_bow.get(
            token, emb_dict_glove_bow['#UNK#']) for token in tokens]
        word_vecs = np.array(word_vecs)
        word_vecs_tensor = torch.tensor(word_vecs)
        sent_vec = word_vecs_tensor.sum(0)
        sent_vec = torch.div(sent_vec, len(tokens))
        bow_glove[q] = sent_vec
    return bow_glove

    
def get_random_embeddings(vocab_size, embedding_size):
    embeddings = np.random.uniform(-1, 1, (vocab_size, embedding_size))
    return embeddings


def get_pretrained_embeddings(glove_path, embedding_size):
    embeddings_glove_bow = []
    emb_dict_glove_bow = {}
    with open(glove_path,'rt', encoding='utf-8') as glove_file:
        pretrained_vectors = glove_file.read().strip().split('\n')
    for vector in range(len(pretrained_vectors)):
        word = pretrained_vectors[vector].split("\t")[0]
        word_vector = [float(val) for val in pretrained_vectors[vector].split('\t')[1].split(' ')[0:]]
        embeddings_glove_bow.append(word_vector)
        emb_dict_glove_bow[word] = word_vector
    vector = 0
    embeddings = np.zeros((len(emb_dict_glove_bow)+2, embedding_size)) 
    for word in emb_dict_glove_bow.keys():
        embeddings[vector] = emb_dict_glove_bow[word]
        vector = vector + 1 
    return embeddings



def class_encoder(labels):
    label_to_int = {}
    for label in labels:
        if label not in label_to_int:
            label_to_int[label] = len(label_to_int)
    return [label_to_int[label] for label in labels]
