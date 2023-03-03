import os
import random

import numpy as np
import torch
from bilstm_classifier import BiLSTMClassifier
from utils import dataset_length, parse_dataset, batch_prediction, STOP_WORDS
import yaml
from bow_classifier import BoWClassifier
from ensemble_classifier import EnsembleClassifier
from torch.utils.data import DataLoader
from dataset_mapper import DatasetMaper
import pickle

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)  
    torch.cuda.manual_seed_all(1234)


def split_training_dataset(training_dataset, split_fraction, training_datapath, validation_datapath):
    # Split the given data set into training and vaidation sets with the given split condition
    with open(training_dataset, "r", encoding="ISO-8859-1") as f:
        contents = f.readlines()
    split_idx = int(len(contents) * split_fraction)
    random.shuffle(contents)
    dev_data = contents[:split_idx]
    train_data = contents[split_idx:]
    with open(validation_datapath, "w", encoding="ISO-8859-1") as f:
        f.writelines(dev_data)
    with open(training_datapath, "w", encoding="ISO-8859-1") as f:
        f.writelines(train_data)


def class_encoder(labels):
    coarse_cls_idx = {}
    idx = 0
    for w in set(labels):
        coarse_cls_idx[w] = idx
        idx = idx + 1
    return [coarse_cls_idx.get(cls) for cls in labels]


def batch_training_bow(epochs,  batch_size, train_sent_vec_list, labels_encoded, learning_rate, model):
    training_set = DatasetMaper(train_sent_vec_list, labels_encoded)
    loader_training = DataLoader(training_set, batch_size=batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):   
        for i, (inputs, labels) in enumerate(loader_training):  
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, len(loader_training), loss.item()))


def get_pretrained_embeddings(glove_path, remove_stopwords):
    embeddings_glove_bow = []
    emb_dict_glove_bow = {}
    with open(glove_path,'rt', encoding='utf-8') as glove_file:
        pretrained_vectors = glove_file.read().strip().split('\n')
    for vector in range(len(pretrained_vectors)):
        word = pretrained_vectors[vector].split("\t")[0]
        word_vector = [float(val) for val in pretrained_vectors[vector].split('\t')[1].split(' ')[0:]]
        if(remove_stopwords):
            if(word not in STOP_WORDS):
                embeddings_glove_bow.append(word_vector)
                emb_dict_glove_bow[word] = word_vector
        else:
            embeddings_glove_bow.append(word_vector)
            emb_dict_glove_bow[word] = word_vector
    return emb_dict_glove_bow, embeddings_glove_bow


def batch_training_bilstm(epochs, batch_size, train_parsed_list, word_to_idx, classes, learning_rate, model, max_sequence_length):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    for epoch in range(epochs):
        for i in range(0, len(train_parsed_list), batch_size):
            batch = train_parsed_list[i:i+batch_size]
            inputs = [[word_to_idx[word] for word in sentence.lower().split()] for sentence, _ in batch]
            inputs = [x + [word_to_idx["#PAD#"]]*(max_sequence_length-len(x)) for x in inputs]
            targets = torch.LongTensor([classes.index(cls) for _, cls in batch])
            optimizer.zero_grad()
            outputs = model(torch.LongTensor(inputs))
            loss = torch.nn.CrossEntropyLoss()(outputs, targets) 
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch+1}/{epochs} | Batch: {i//batch_size+1}/{len(train_parsed_list)//batch_size+1} | Loss: {loss.item():.4f}")


def persist_configurations(model, word_indices, train_classes, training_max_sentence_length, config):
    with open(config["model_path"], "wb") as bilstm_path:
            pickle.dump(model, bilstm_path, protocol=pickle.DEFAULT_PROTOCOL)

    with open(config["word_indices"], "wb") as word_indices_path:
            pickle.dump(word_indices, word_indices_path, protocol=pickle.DEFAULT_PROTOCOL)

    with open(config["training_classes"], "wb") as training_classes_path:
            pickle.dump(train_classes, training_classes_path, protocol=pickle.DEFAULT_PROTOCOL)

    with open(config["metadata"], "wb") as metadata_path:
            pickle.dump(training_max_sentence_length, metadata_path, protocol=pickle.DEFAULT_PROTOCOL)
    
    print("\nAll configurations persisted successfully for testing\n")

def train(config_file):
    with open(
        os.getcwd().replace("src","data/") + config_file, "r", encoding="ISO-8859-1"
    ) as configurations:
        config = yaml.load(configurations, Loader=yaml.Loader)

    dataset_length(config["original_dataset"], "original")

    split_training_dataset(
        config["original_dataset"],
        config["split_ratio"],
        config["training_dataset"],
        config["validation_dataset"],
    )

    dataset_length(config["training_dataset"], "training")
    dataset_length(config["validation_dataset"], "validation")

    training_question_list, training_coarse_labels, training_fine_labels, training_max_sentence_length, train_parsed_list = parse_dataset(
        config["training_dataset"], config["predict_class"])

    training_coarse_labels_encoded = class_encoder(training_coarse_labels)
    training_fine_labels_encoded = class_encoder(training_fine_labels)

    _, _, _, _, validation_parsed_list = parse_dataset(
        config["validation_dataset"], config["predict_class"])


    # Train & validate the model either on fine or coarse classes
    if(config["predict_class"]=="fine"):
        num_classes = config["fine_classes"]
        train_classes = list(set(training_fine_labels))
        labels_encoded = training_fine_labels_encoded
    else:
        num_classes = config["coarse_classes"]
        train_classes = list(set(training_coarse_labels))
        labels_encoded = training_coarse_labels_encoded


# **************************************************** BoW begins ******************************************************************
    if(config["model"] == "bow"):
        print("\nIt's Bag-Of-Words")
        if(config["embedding"] == "random"):
            custom_vocab_list = []
            custom_vocab_count = {}
            word_indices = {
                "#PAD#": 0,
                "#UNK#": 1,
            } 
            for ques in training_question_list:
                custom_vocab_list = custom_vocab_list + ques.lower().split(' ')

            if(config["remove_stopwords"]):
                custom_vocab_list = [i for i in custom_vocab_list if i not in STOP_WORDS]

            for i in custom_vocab_list:
                if i not in list(custom_vocab_count.keys()):
                    custom_vocab_count[i] = 1
                else:
                    custom_vocab_count[i] = custom_vocab_count[i] + 1
            custom_vocab_count_frequency = {k: v for k, v in custom_vocab_count.items() if v >= config["min_freq"]}
            idx_counter = 2
            for w in custom_vocab_count_frequency.keys():
                word_indices[w] = idx_counter
                idx_counter = idx_counter + 1
            sent_idx_random = []
            for q in training_question_list:
                tokens = q.lower().split()
                sent_indices = [word_indices.get(token, word_indices["#UNK#"]) for token in tokens]    
                if len(sent_indices) < training_max_sentence_length:
                    sent_indices += [word_indices["#PAD#"]] * (training_max_sentence_length - len(sent_indices))
                sent_indices_tensor = torch.tensor(sent_indices)
                sent_idx_random.append(sent_indices_tensor)
            embeddings = torch.randn(len(word_indices), config["word_embedding_dim"])
        else:
            emb_dict_glove_bow, embeddings = get_pretrained_embeddings(config["pretrained_glove"], config["remove_stopwords"])
            zeroes = [0 for i in range(0, 300)]
            embeddings.append(zeroes)
            word_indices = {}
            idx_counter = 0
            sent_idx_random = []

            for word in list(emb_dict_glove_bow.keys()):
                word_indices[word] = idx_counter
                idx_counter = idx_counter + 1
            word_indices["#PAD#"] = idx_counter
            
            for q in training_question_list:
                tokens = q.lower().split()
                sent_indices = [word_indices.get(token, word_indices["#UNK#"]) for token in tokens]    
                if len(sent_indices) < training_max_sentence_length:
                    sent_indices += [word_indices["#PAD#"]] * (training_max_sentence_length - len(sent_indices))
                sent_idx_random.append(sent_indices)
            sent_idx_random = torch.tensor(sent_idx_random)


    # ***************************************** BoW model training begins ****************************************************

        model = BoWClassifier(config["word_embedding_dim"], config["hidden_dim"], num_classes,
                                  embeddings, word_indices["#PAD#"], config["freeze_embeddings"])

        print("\nTraining the model...")
        batch_training_bow(config["epochs"], config["batch_size"], sent_idx_random,
                            labels_encoded, config["learning_rate"], model)

    # ***************************************** BoW model training ends ******************************************************

# **************************************************** BoW ends ******************************************************************



# ************************************************* BiLSTM begins ****************************************************************
    elif(config["model"] == "bilstm"):
        print("\nIt's BiLSTM")
        vocab = set([word for sentence, _  in train_parsed_list for word in sentence.lower().split()])

        if(config["remove_stopwords"]):
            vocab = vocab - set(STOP_WORDS)
            
        word_indices = {word: index + 2 for index, word in enumerate(vocab)}
        word_indices["#PAD#"] = 0
        word_indices["#UNK#"] = 1
        vocab_size = len(word_indices)

        if(config["embedding"] == "random"):
            embeddings = np.random.uniform(-1, 1, (vocab_size, config["word_embedding_dim"]))
        else:
            emb_dict_glove_bow, _ = get_pretrained_embeddings(config["pretrained_glove"], config["remove_stopwords"])
            embeddings = np.zeros((len(emb_dict_glove_bow)+2, config["word_embedding_dim"])) 
            i = 0
            for word in emb_dict_glove_bow.keys():
                embeddings[i] = emb_dict_glove_bow[word]
                i = i + 1 

        
        # ***************************************** BiLSTM model training begins **************************************************

        model = BiLSTMClassifier(config["word_embedding_dim"], config["hidden_dim"], num_classes,
                                 torch.FloatTensor(embeddings), config["freeze_embeddings"])

        print("\nTraining the model...")
        batch_training_bilstm(config["epochs"], config["batch_size"], train_parsed_list, word_indices,
                              train_classes, config["learning_rate"], model, training_max_sentence_length)
        
    # ***************************************** BiLSTM model training ends *****************************************************

# **************************************************** BiLSTM ends *************************************************************



# *************************************************** Ensemble begins **********************************************************
    else:
        print("It's ensemble learning!")
        vocab = set([word for sentence, _  in train_parsed_list for word in sentence.lower().split()])    
        word_indices = {word: index + 2 for index, word in enumerate(vocab)}
        word_indices["#PAD#"] = 0
        word_indices["#UNK#"] = 1
        vocab_size = len(word_indices)

        emb_dict_glove_bow, _ = get_pretrained_embeddings(config["pretrained_glove"], config["remove_stopwords"])
        embeddings = np.zeros((len(emb_dict_glove_bow)+2, config["word_embedding_dim"])) 
        i = 0
        for word in emb_dict_glove_bow.keys():
            embeddings[i] = emb_dict_glove_bow[word]
            i = i + 1 

        model = EnsembleClassifier(config["word_embedding_dim"], config["hidden_dim"], num_classes,
                                 torch.FloatTensor(embeddings), config["freeze_embeddings"], word_indices["#PAD#"])

        print("\nTraining the model...")
        batch_training_bilstm(config["epochs"], config["batch_size"], train_parsed_list, word_indices,
                              train_classes, config["learning_rate"], model, training_max_sentence_length)

# *************************************************** Ensemble ends **********************************************************

    print("\nValidating the model...") 
    batch_prediction(validation_parsed_list, word_indices, model, train_classes, training_max_sentence_length, "validation")

    persist_configurations(model, word_indices, train_classes, training_max_sentence_length, config)