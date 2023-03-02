import os

import yaml
from utils import dataset_length, parse_dataset, batch_prediction
import torch
import pickle
import random
import numpy as np

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)  
    torch.cuda.manual_seed_all(1234)

def test(config_file):
    with open(os.getcwd().replace("src","data/") + config_file, "r", encoding="ISO-8859-1") as configurations:
        config = yaml.load(configurations, Loader=yaml.Loader)

    dataset_length(config["test_dataset"], "testing")

    _, _, _, _, testing_parsed_list = parse_dataset(
        config["test_dataset"], config["predict_class"])

    with open(config["model_path"], "rb") as model_path:
            model =  pickle.load(model_path)

    with open(config["word_indices"], "rb") as word_indices_path:
            word_indices =  pickle.load(word_indices_path)

    with open(config["training_classes"], "rb") as training_classes_path:
            training_classes =  pickle.load(training_classes_path)

    with open(config["metadata"], "rb") as metadata_path:
        training_max_sentence_length =  pickle.load(metadata_path)



    batch_prediction(testing_parsed_list, word_indices, model, training_classes, training_max_sentence_length, "testing")

