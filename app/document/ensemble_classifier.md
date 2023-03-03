# Ensemble Classifier

This code defines a PyTorch neural network model called EnsembleClassifier. The model is defined by inheriting from the PyTorch Module class. The constructor method of the EnsembleClassifier class takes in several arguments:
- input_size: an integer representing the size of the input layer.
- hidden_size: an integer representing the size of the hidden layer in the LSTM component of the model.
- output_size: an integer representing the size of the output layer (i.e., the number of classes for sentiment analysis).
- embeddings: a numpy array containing pre-trained word embeddings.
- freeze: a boolean flag indicating whether the embeddings should be frozen during training.
- padding_token: an integer representing the index of the padding token in the embeddings array.

In the <b>__init__</b> method, the model creates several layers:
- an embedding layer, which uses pre-trained word embeddings to convert text data into vectors of fixed size
- a bidirectional LSTM layer, which processes the embedded input and generates a hidden representation of the text
- a linear layer that maps the LSTM hidden representation to the output layer
 - a linear layer that maps the bag-of-words (BoW) representation of the text to the output layer

The <b>forward</b> method of the model performs the following steps: 
- The input tensor is first passed through the embedding layer to obtain a tensor of embedded vectors
- The LSTM layer processes the embedded vectors and generates a hidden representation of the text
- The last hidden state of the LSTM is used as the LSTM score
- Additionally, the embedded vectors are averaged to obtain a bag-of-words (BoW) representation of the text, which is passed through a linear layer to obtain the BoW score
- Finally, the BoW score and LSTM score are averaged to obtain the final scores for each class. The output is returned as a tensor.

>**Note:** The code also sets the random seed for reproducibility and checks if a GPU is available to use.