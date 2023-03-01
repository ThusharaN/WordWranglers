# BiLSTMClassifier

<b>BiLSTMClassifier</b> class, a subclass of <b>nn.Module</b>, is used to define a neural network model that acts as a question classifier, specifically when BiLSTM is used to generate the sentence representation

The <i>init</i> method that takes four arguments:
- input_dim: the dimension of the embeddings produced by BiLSTM
- hidden_dim: the dimension of the hidden state of the LSTM layer
- output_dim: the dimension of the output features; 6 for coarse labels, 50 for fine labels
- embeddings: the pre-trained embedding matrix produced by BiLSTM
- freeze_embeddings: a boolean value indicating whether to freeze the embedding layer during training

The class also has a forward method that takes a batch of input sequences as an argument and returns the predicted output for each sequence in the batch.

The <i>forward method</i>, that the input sequence and returns the predicted output for each sequence in the batch, performs the following operations:
- The input sequences are embedded using the pre-trained embedding matrix
- This embedded input is passed through a bidirectional LSTM layer
- The last hidden state of the forward and backward LSTMs is concatenated
- A ReLU activation function is applied to the concatenated hidden state
- The resulting vector is passed through the final complete linear layer to obtain the output.