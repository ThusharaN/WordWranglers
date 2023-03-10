import random
import torch
import numpy as np
from sklearn.metrics import f1_score
import torch.nn.functional as F
import string


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)  
    torch.cuda.manual_seed_all(1234)


PUNCTUATIONS = [punct for punct in string.punctuation]


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
        for punctuation in PUNCTUATIONS:
            question = question.replace(punctuation, "")
        question_list.append(question)
        coarse_class_labels.append(coarse_class)
        fine_class_labels.append(fine_class)
        if(prediction_class == "fine"):
            parsed_list.append((question, fine_class))
        else:
            parsed_list.append((question, coarse_class))
    print(f"Longest sentence in file {filename} is of length: {max_sentence_length}")
    return question_list, coarse_class_labels, fine_class_labels, max_sentence_length, parsed_list


def batch_prediction(validation_parsed_list, word_to_idx, model, classes, max_sequence_length, mode):
    model.eval()
    batch_size = 1
    pred_cls_list = []
    true_cls_list = []
    with torch.no_grad():
        num_correct = 0
        total_loss = 0
        for i in range(0, len(validation_parsed_list), batch_size):
            batch = validation_parsed_list[i:i+batch_size]
            inputs = [[word_to_idx.get(word, word_to_idx["#UNK#"]) for word in sentence.lower().split()] for sentence, _ in batch]
            inputs = [x + [word_to_idx["#PAD#"]]*(max_sequence_length-len(x)) for x in inputs]
            output = model(torch.LongTensor(inputs))
            ot = F.softmax(output, dim = 1).argmax(dim = 1)
            pred_cls = classes[ot]
            targets = torch.LongTensor([classes.index(cls) for _, cls in batch])
            loss = torch.nn.CrossEntropyLoss()(output, targets)
            total_loss += loss.item()
            if pred_cls == validation_parsed_list[i][1]:
                num_correct += 1
            print(f"Sentence: {validation_parsed_list[i][0]} | Predicted class: {pred_cls} | True class: {validation_parsed_list[i][1]}")
            pred_cls_list.append(pred_cls)
            true_cls_list.append(validation_parsed_list[i][1])
    print(f"\nAccuracy during {mode} is: {num_correct / len(validation_parsed_list):.2f}")
    print(f"\nAverage Loss during {mode} is: {total_loss / len(validation_parsed_list):.2f}")
    print(f"\nF1 score (macro) during {mode} is: {f1_score(true_cls_list, pred_cls_list, average='macro'):.2f}")
