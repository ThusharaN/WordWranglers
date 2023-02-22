import string
from collections import Counter

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
] + [punct for punct in string.punctuation]


def dataset_length(dataset, type):
    # Verifying the given dataset
    with open(dataset, "r") as file:
        line_count = 0
        for line in file:
            line_count += 1
    print(f"Number of lines in {type} dataset: {line_count}")


def parse_dataset(filename):
    # Parsing the given dataset tp return list of questions, coarse and fine classes
    dataset = open(filename, "r")
    question_list = []
    fine_class_labels = []
    coarse_class_labels = []
    max_sentence_length = 0
    tagged_list = []
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
        tagged_list.append((question, coarse_class, fine_class))
    print(f"Maximum sentence: {max_sentence_length}")
    return tagged_list, question_list, coarse_class_labels, fine_class_labels


def get_vocab(dataset, vocab_size, min_freq):
    vocab = {
        "<PAD>": 0,  # special token used for padding sequences
        "<UNK>": 1,  # special token used for out-of-vocabulary words
    }
    word_counts = Counter(
        token
        for question, coarse_class, fine_class in dataset
        for token in question.lower().split(" ")
        if token not in STOP_WORDS
    )
    for i, (word, count) in enumerate(
        word_counts.most_common(vocab_size - 2)
    ):
        if count >= min_freq:
            vocab[word] = i + 2
    print(f"Vocabulary length {len(vocab)}")
    return vocab
