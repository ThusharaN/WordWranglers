# Model Testing

This script tests the performance of the text classification model trained in the earlier steps. The steps involved in the testing process are as follows:
- The <i><b>test</b></i> function reads the configuration file specified in the given argument using the yaml.load method from the yaml package. This configuration file contains paths to the test dataset, the saved model, the word indices, the training classes, and the metadata.
- Next, the function calls the <i>dataset_length</i> and <i>parse_dataset</i> functions from utilities module to get the length of the testing dataset and parse the dataset into a list of questions, coarse and fine classes.
- The saved model, word indices, training classes, and other metadata are then loaded from their respective paths using Python'sin-built pickle module
- Finally, the <i>batch_prediction</i> function from the utilities module is called to predict the classes of the questions in the testing dataset using the loaded model and word indices. The function prints the predicted and true classes for each sentence in the testing dataset, as well as the F1 scores for the testing dataset.


>**Note:** The code also sets the random seed for reproducibility and checks if a GPU is available to use.