# QuestionDataset

<b>QuestionDataset</b>, inherited from <b>torch.utils.data.Dataset</b> is used to represent the questions in the training/validation/test dataset and their corresponding labels (fine or coarse), where each question is represented by a dictionary of indices for each word (word2idx).

The <i>__init__</i> method initializes the dataset with the following parameters:

- questions: a list of tokenized questions represented as integer sequences
- labels: a list of corresponding labels for each question
- word2idx: a dictionary that maps words to their corresponding integer indices
- max_seq_len: the maximum length of each question sequence

The <i>__len__</i> method overrides the in-built <i>len</i> method to return the number of questions as the length of the QuestionDataset.

The <i>__getitem__</i> method returns the i-th question and label from the questions and labels lists, respectively, as a tuple which is later used by PyTorch's data loaders to retrieve batches of data during training or evaluation.