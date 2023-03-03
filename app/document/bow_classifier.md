# BoW Classifier

<b>BoWClassifier</b> is a neural network model that uses bag of words approach to classify text data. The class inherits from the <b>nn.Module</b> class which allows it to use all the functionalities of the PyTorch framework.

The __init__ function initializes the layers of the neural network. It takes in four arguments:
- input_size is the size of the input layer which is the number of unique words in the vocabulary.
- hidden_size is the size of the hidden layer.
- output_size is the number of output classes (6 corase classes & 50 fine classes)
- embeddings is a pre-trained embedding matrix that is used to convert the input text into vector representations.
- padding_token is the index of the padding token <i>#PAD#</i> in the embedding matrix.

The layers of the neural network include:
- nn.Embedding: This layer takes the input of the pre-trained embedding matrix and returns a tensor where each element in the input tensor is replaced by its corresponding vector representation from the embedding matrix.
- nn.Linear: This is a fully connected layer that takes the input tensor and applies a linear transformation to it.
- nn.ReLU: This is a non-linear activation function that applies the rectified linear unit function to the output of the first linear layer.

The <b>forward</b> function defines how the input data is propagated through the neural network. It takes in an input tensor and returns an output tensor.
- The input tensor is first passed through the nn.Embedding layer to be converted into vector representations
- Next, a list is created to store the mean of the embeddings of each sentence in the input tensor
- The list is then converted to a tensor resulting in a tensor of shape (batch_size, embedding_size)
- The tensor is then passed through the linear layers defined in <i>__init__</i> in order: nn.Linear, nn.ReLU, nn.Linear
- The output tensor has shape (batch_size, output_size) and represents the predicted probabilities of the input belonging to each of the output classes.

>**Note:** The code also sets the random seed for reproducibility and checks if a GPU is available to use.