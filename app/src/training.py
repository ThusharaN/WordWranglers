import os
import random

import torch
from app.src.utils import dataset_length, parse_dataset, get_randomly_initialised_bow, get_pre_trained_bow
import yaml
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


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


def train(config_file):
    torch.manual_seed(1)
    random.seed(1)

    with open(
        os.getcwd() + "/app/data/" + config_file, "r", encoding="ISO-8859-1"
    ) as configurations:
        config = yaml.load(configurations, Loader=yaml.Loader)

    dataset_length(
        config["original_dataset"],
        "train")

    split_training_dataset(
        config["original_dataset"],
        config["split_ratio"],
        config["training_dataset"],
        config["validation_dataset"],
    )

    training_question_list, training_coarse_labels, training_fine_labels, training_max_sentence_length = parse_dataset(
        config["training_dataset"])
    validation_question_list, validation_coarse_labels, validation_fine_labels, validation_max_sentence_length = parse_dataset(
        config["validation_dataset"])
    max_seq_length = training_max_sentence_length

    # Creating a sentence representation based on the supplied configuration
    sentence_rep = {}
    if(config["model"] == "bow"):
        if(config["embedding"] == "random"):
            sentence_rep = get_randomly_initialised_bow(
                training_question_list, config["min_freq"], config["word_embedding_dim"], max_seq_length, True)
            print(sentence_rep)
        else:
            sentence_rep = get_pre_trained_bow(
                training_question_list, config["pretrained_glove"], True)
    else:
        print("It's bilstm")

    # TODO: Validation data
    # test_data_list = parse_dataset(
    #     constants.TESTING_DATA, question_list, fine_class_labels, coarse_class_labels, max_sentence_length)
    # dataset_length(constants.TESTING_DATA, "test")
