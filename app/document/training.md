# Model Training

This file defines a series of functions to train a text classifier. The functions defined as part of training the classifier are as follows:

<b><i>split_training_dataset</i></b>
- Takes a path to a training dataset, a split fraction, and paths to the output training and validation files 
- Splits the training dataset into two parts according to the split fraction and saves them to the specified paths

<b><i>class_encoder</i></b>
- Takes a list of labels and returns a list of indices that encode the classes in a numerical form

<b><i>batch_training_bow</i></b> 
- Trains a bag-of-words classifier on the given input data, using a specified number of epochs, batch size, learning rate, and the supplied model
- Creates a DataLoader object to efficiently load batches of data during training and uses the Adam optimizer and Cross Entropy Loss criterion to optimize the model's parameters
- Loops through each epoch, running batches of data through the model and optimizing the parameters via backpropagation 

<b><i>get_pretrained_embeddings</i></b>
- Loads pre-trained GloVe embeddings from the given file and returns them as a dictionary of word vectors and a list of embedding vectors
- Initializes two empty lists which will store the pre-trained embeddings and the corresponding words, respectively
- Reads in the pre-trained vectors from the given file path and splits them, looping over each vector to extract the word and corresponding vector and appending them to the list and the dictionary, respectively
- Gets reused for both Bag-Of-Words and BiLSTM based classifiers when the classifier is expected to use pretrained embeddings.

<b><i>batch_training_bilstm</i></b> 
- Trains a bidirectional LSTM (BiLSTM) classifier on the given input data, using a specified number of epochs, batch size, learning rate, model, and maximum sequence length used to deicde the amount of apdding needed
- CrossEntropy loss is calculated between the predictions and target indices, and the gradients are backpropagated through the network
- The Optimizer is stepped and the loss is printed for monitoring purposes


<b><i>persist_configurations</i></b>
- Saves the trained model and other metadata to the disk, according to the specified configuration file
- Persisted configurations later retrieved for testing the trained model

<b><i>train</i></b>
- Main entry point for the script
- Loads the configuration file specified by the supplied argument, reads in the training data, trains the specified model, and saves the trained model and metadata to the disk
- Calls the functions mentioned earlier to complete the training and validation of the classifier and persist the configurations

>**Note:** The code also sets the random seed for reproducibility and checks if a GPU is available to use.