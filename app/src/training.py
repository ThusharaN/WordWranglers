import os
import random

import numpy as np
import torch
from app.src.bilstm_classifier import BiLSTMClassifier
from app.src.dataset import QuestionDataset
from app.src.utils import dataset_length, parse_dataset, get_randomly_initialised_bow, get_pre_trained_bow, class_encoder, get_randomly_initialised_bilstm
import yaml
from app.src.bow_classifier import Classifier
from torch.utils.data import DataLoader


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

def model_preprocessing_bow(question_list, coarse_labels_encoded, fine_labels_encoded, seq_length, batch_size, shuffle):
    words = [word for question in question_list for word in question.split()]
    word2idx = {word: idx for idx, word in enumerate(set(words))}
    word2idx["#PAD#"] = len(word2idx)
    word2idx["#UNK#"] = len(word2idx)
    padded_questions = [question + " #PAD#" * (seq_length - len(
        question.split())) for question in question_list]
    padded_questions = list(set(padded_questions))
    num_words = len(word2idx)
    # num_coarse_classes = len(set(coarse_labels_encoded))
    # num_fine_classes = len(set(fine_labels_encoded))
    dataset = QuestionDataset(padded_questions, coarse_labels_encoded, fine_labels_encoded,  word2idx, seq_length)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=shuffle)
    return num_words, dataloader, dataset

def batch_training_bow(sentence_rep, dataloader, optimizer, model, epochs):
    for epoch in range(epochs):
        for batch_idx, (questions, coarse_labels, fine_labels) in enumerate(dataloader):
            batch_sentence_rep = {key: sentence_rep[key.replace(" #PAD#", "")] for key in questions if isinstance(key, str)}
            question_tensor = torch.stack(
                [v.abs().long() for v in batch_sentence_rep.values() if isinstance(v, torch.Tensor)])
            optimizer.zero_grad()
            coarse_output, fine_output = model(question_tensor)
            coarse_loss = torch.nn.functional.cross_entropy(
                coarse_output, coarse_labels)
            fine_loss = torch.nn.functional.cross_entropy(
                fine_output, fine_labels)
            loss = coarse_loss + fine_loss
            loss.backward()
            optimizer.step()
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                .format(epoch + 1, epochs, batch_idx + 1, len(dataloader), loss.item()))

def batch_validation_bow(test_dataloader, test_dataset, validation_sentence_rep, model):
    coarse_preds = []
    fine_preds = []
    for _, (_, _, _) in enumerate(test_dataloader):
        question_tensor = torch.stack(
            [v.abs().long() for v in validation_sentence_rep.values() if isinstance(v, torch.Tensor)])
        with torch.no_grad():
            coarse_output, fine_output = model(question_tensor)
            coarse_pred = coarse_output.argmax(dim = 1)
            fine_pred = fine_output.argmax(dim = 1)
        coarse_preds.extend(coarse_pred.tolist())
        fine_preds.extend(fine_pred.tolist())

    coarse_correct = sum([1 for i in range(len(test_dataset)) if coarse_preds[i] == test_dataset[i][1]])
    fine_correct = sum([1 for i in range(len(test_dataset)) if fine_preds[i] == test_dataset[i][2]])
    coarse_acc = coarse_correct / len(test_dataset)
    fine_acc = fine_correct / len(test_dataset)
    return coarse_acc, fine_acc

def batch_training_bilstm(epochs, batch_size, train_parsed_list, word_to_idx, training_coarse_labels, optimizer, model):
    for epoch in range(epochs):
        for i in range(0, len(train_parsed_list), batch_size):
            batch = train_parsed_list[i: i + batch_size]
            inputs = [[word_to_idx[word] for word in sentence.lower().split()] for sentence, _ in batch]
            #max_len = max([len(x) for x in inputs])
            inputs = [x + [0]*(40-len(x)) for x in inputs]
            targets = torch.LongTensor([list(set(training_coarse_labels)).index(cls) for _, cls in batch])
            optimizer.zero_grad()
            outputs = model(torch.LongTensor(inputs))
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch+1}/{epochs} | Batch: {i//batch_size+1}/{len(train_parsed_list)//batch_size+1} | Loss: {loss.item():.4f}")

def batch_validation_bilstm(test_parsed_list, batch_size, word_to_idx, unknown_token, model, classes):
    with torch.no_grad():
        num_correct = 0
        for i in range(0, len(test_parsed_list), batch_size):
            batch = test_parsed_list[i:i+batch_size]
            inputs = [[word_to_idx.get(word, word_to_idx[unknown_token]) for word in sentence.lower().split()] for sentence, _ in batch]
            inputs = [x + [0]*(40-len(x)) for x in inputs]
            output = model(torch.LongTensor(inputs))
            ot = output.argmax().item()
            pred_cls = classes[ot]
            if pred_cls == test_parsed_list[i][1]:
                num_correct += 1
            print(f"Sentence: {test_parsed_list[i][0]} | Predicted class: {pred_cls} | True class: {test_parsed_list[i][1]}")
        accuracy = num_correct / len(test_parsed_list)
    return accuracy

def train(config_file):
    torch.manual_seed(1)
    random.seed(1)

    with open(
        os.getcwd() + "/app/data/" + config_file, "r", encoding="ISO-8859-1"
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
        config["training_dataset"])
    training_coarse_labels_encoded = class_encoder(training_coarse_labels)
    training_fine_labels_encoded = class_encoder(training_fine_labels)

    validation_question_list, validation_coarse_labels, validation_fine_labels, validation_max_sentence_length, validation_parsed_list = parse_dataset(
        config["validation_dataset"])
    validation_coarse_labels_encoded = class_encoder(validation_coarse_labels)
    validation_fine_labels_encoded = class_encoder(validation_fine_labels)

    training_max_seq_length = training_max_sentence_length

    validation_max_seq_length = validation_max_sentence_length

    # Creating a sentence representation based on the supplied configuration
    sentence_rep = {}
    if(config["model"] == "bow"):
        print("It's Bag-Of-Words")
        if(config["embedding"] == "random"):
            sentence_rep = get_randomly_initialised_bow(training_question_list, config["min_freq"], config["word_embedding_dim"], training_max_seq_length, True)
            validation_sentence_rep = get_randomly_initialised_bow(validation_question_list, config["min_freq"], config["word_embedding_dim"], validation_max_seq_length, True)
        else:
            sentence_rep = get_pre_trained_bow(training_question_list, config["pretrained_glove"], True)
            validation_sentence_rep = get_pre_trained_bow(validation_question_list, config["pretrained_glove"], True)

        print("Training the model...")   
        num_words, dataloader, _ = model_preprocessing_bow(training_question_list, training_coarse_labels_encoded,
                                                                training_fine_labels_encoded, training_max_seq_length, config["batch_size"], True)
        model = Classifier(num_words, config["word_embedding_dim"], config["coarse_classes"], config["fine_classes"])
        optimizer = torch.optim.Adam(model.parameters())
        batch_training_bow(sentence_rep, dataloader, optimizer, model, config["epochs"])

        print("Validating the model...")  
        model.eval()
        _, test_dataloader, test_dataset = model_preprocessing_bow(validation_question_list, validation_coarse_labels_encoded,
                                                                          validation_fine_labels_encoded, validation_max_seq_length, config["batch_size"], False)
        coarse_acc, fine_acc = batch_validation_bow(test_dataloader, test_dataset, validation_sentence_rep, model)
        print('Coarse class accuracy: {:.2%}, Fine class accuracy: {:.2%}'.format(coarse_acc, fine_acc))
    else:
        print("It's BiLSTM")
        if(config["embedding"] == "random"):
            word_to_idx, random_embeddings_bilstm = get_randomly_initialised_bilstm(train_parsed_list, config["word_embedding_dim"])
        else:
            print("GloVe implementation - TBD")

        print("Training the model...") 
        model = BiLSTMClassifier(config["word_embedding_dim"], config["hidden_dim"], config["output_dim"],
                                 torch.FloatTensor(random_embeddings_bilstm), config["freeze_embeddings"])
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        batch_training_bilstm(config["epochs"], config["batch_size"], train_parsed_list, 
         word_to_idx, training_coarse_labels, optimizer, model)

        print("Validating the model...") 
        accuracy = batch_validation_bilstm(validation_parsed_list, 1, word_to_idx, 
        "#UNK#", model, list(set(validation_coarse_labels)))
        print(f"Accuracy: {accuracy:.2f}")

        