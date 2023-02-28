import os
import random

import numpy as np
import torch.nn.functional as F
import torch
from app.src.bilstm_classifier import BiLSTMClassifier
from app.src.dataset import QuestionDataset
from app.src.utils import dataset_length, parse_dataset, get_randomly_initialised_bow, get_pre_trained_bow, class_encoder, get_random_embeddings, get_pretrained_embeddings
import yaml
from app.src.bow_classifier import Classifier
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score 

torch.manual_seed(1)
torch.cuda.manual_seed(1)                                                                                                                              
torch.cuda.manual_seed_all(1)                                                                                          
np.random.seed(1)                                                                                                             
random.seed(1) 

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

def model_preprocessing_bow(question_list, labels_encoded, seq_length, batch_size, shuffle):
    words = [word for question in question_list for word in question.split()]
    word2idx = {word: idx+2 for idx, word in enumerate(set(words))}
    word2idx["#PAD#"] = 0
    word2idx["#UNK#"] = 1
    padded_questions = [question + " #PAD#" * (seq_length - len(
        question.split())) for question in question_list]
    padded_questions = list(set(padded_questions))
    num_words = len(word2idx)+2
    dataset = QuestionDataset(padded_questions, labels_encoded,  word2idx, seq_length)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=shuffle)
    return num_words, dataloader, dataset

def batch_training_bow(sentence_rep, dataloader, epochs,  word_embedding_dim, hidden_dim, classes):
    for epoch in range(epochs):
        for batch_idx, (questions, labels) in enumerate(dataloader):
            batch_sentence_rep = {key: sentence_rep[key.replace(" #PAD#", "")] for key in questions if isinstance(key, str)}
            question_tensor = [v.tolist() for v in batch_sentence_rep.values()]
            question_tensor = np.array(question_tensor)
            model = Classifier(question_tensor, word_embedding_dim, hidden_dim, classes)
            
            #optimizer = torch.optim.Adam(model.parameters())
            output = model(question_tensor)
            #optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(output, labels)
            loss.backward()
            #optimizer.step()
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                .format(epoch + 1, epochs, batch_idx + 1, len(dataloader), loss.item()))
    return model

def batch_validation_bow(test_dataloader, test_dataset, validation_sentence_rep, model):
    preds = []
    for _, (_, _,) in enumerate(test_dataloader):
        question_tensor = torch.stack(
            [v for v in validation_sentence_rep.values() if isinstance(v, torch.Tensor)])
        with torch.no_grad():
            output = model(question_tensor)
            pred = F.softmax(output, dim=1).argmax(dim=1)
            # pred = output.argmax(dim = 1)
        preds.extend(pred.tolist())
    correct = sum([1 for i in range(len(test_dataset)) if preds[i] == test_dataset[i][1]])
    accuracy = correct / len(test_dataset)
    return accuracy

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


def batch_validation_bilstm(validation_parsed_list, word_to_idx, model, classes, max_sequence_length):
    batch_size = 1
    pred_cls_list = []
    true_cls_list = []
    with torch.no_grad():
        num_correct = 0
        for i in range(0, len(validation_parsed_list), batch_size):
            batch = validation_parsed_list[i:i+batch_size]
            inputs = [[word_to_idx.get(word, word_to_idx["#UNK#"]) for word in sentence.lower().split()] for sentence, _ in batch]
            inputs = [x + [word_to_idx["#PAD#"]]*(max_sequence_length-len(x)) for x in inputs]
            output = model(torch.LongTensor(inputs))
            ot = F.softmax(output, dim = 1).argmax(dim = 1)
            pred_cls = classes[ot]
            if pred_cls == validation_parsed_list[i][1]:
                num_correct += 1
            print(f"Sentence: {validation_parsed_list[i][0]} | Predicted class: {pred_cls} | True class: {validation_parsed_list[i][1]}")
            pred_cls_list.append(pred_cls)
            true_cls_list.append(validation_parsed_list[i][1])
        print(f"\nF1 score with the validation set is: {f1_score(true_cls_list, pred_cls_list, average='micro'):.2f}")

def train(config_file):
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
        config["training_dataset"], config["predict_class"])
    training_coarse_labels_encoded = class_encoder(training_coarse_labels)
    training_fine_labels_encoded = class_encoder(training_fine_labels)

    validation_question_list, validation_coarse_labels, validation_fine_labels, validation_max_sentence_length, validation_parsed_list = parse_dataset(
        config["validation_dataset"], config["predict_class"])
    validation_coarse_labels_encoded = class_encoder(validation_coarse_labels)
    validation_fine_labels_encoded = class_encoder(validation_fine_labels)
    
    vocab = set([word for sentence, _  in train_parsed_list for word in sentence.lower().split()])    
    word_indices = {word: index + 2 for index, word in enumerate(vocab)}
    word_indices["#PAD#"] = 0
    word_indices["#UNK#"] = 1
    vocab_size = len(word_indices)

    # training_max_seq_length = training_max_sentence_length
    # validation_max_seq_length = validation_max_sentence_length

    # TODO: Check with training_max_sentence_length & validation_max_sentence_length
    training_max_seq_length = 40
    validation_max_seq_length = 40

    # Train & validate the model either on fine or coarse classes
    if(config["predict_class"]=="fine"):
        num_classes = config["fine_classes"]
        train_classes = list(set(training_fine_labels))
        validation_classes = list(set(validation_fine_labels))
    else:
        num_classes = config["coarse_classes"]
        train_classes = list(set(training_coarse_labels))
        validation_classes = list(set(validation_coarse_labels))

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
        if(config["predict_class"]=="fine"):
            labels_encoded = training_fine_labels_encoded
        else:
            labels_encoded = training_coarse_labels_encoded
        _, dataloader, _ = model_preprocessing_bow(training_question_list, labels_encoded,
                                                           training_max_seq_length, config["batch_size"], True)

        model = batch_training_bow(sentence_rep, dataloader, config["epochs"],  config["word_embedding_dim"], config["hidden_dim"], num_classes)

        print("Validating the model...")
        if(config["predict_class"]=="fine"):
            labels_encoded = validation_fine_labels_encoded
        else:
            labels_encoded = validation_coarse_labels_encoded
        model.eval()
        _, test_dataloader, test_dataset = model_preprocessing_bow(validation_question_list, labels_encoded,
                                                                   validation_max_seq_length, 1, False)
        coarse_acc, fine_acc = batch_validation_bow(test_dataloader, test_dataset, validation_sentence_rep, model)
        print('Coarse class accuracy: {:.2%}, Fine class accuracy: {:.2%}'.format(coarse_acc, fine_acc))
    else:
        print("It's BiLSTM")
        # TODO: Move this code above so that the embeddings can be used for both bow & bilstm
        if(config["embedding"] == "random"):
            embeddings = get_random_embeddings(vocab_size, config["word_embedding_dim"])
        else:
            embeddings = get_pretrained_embeddings(config["pretrained_glove"], config["word_embedding_dim"])

        model = BiLSTMClassifier(config["word_embedding_dim"], config["hidden_dim"], num_classes,
                                 torch.FloatTensor(embeddings), config["freeze_embeddings"])

        print("Training the model...")
        batch_training_bilstm(config["epochs"], config["batch_size"], train_parsed_list, word_indices,
                              train_classes, config["learning_rate"], model, training_max_seq_length)

        print("Validating the model...") 
        batch_validation_bilstm(validation_parsed_list, word_indices, model,
                                train_classes, validation_max_seq_length)

        