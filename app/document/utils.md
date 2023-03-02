 # Utilities
 
This code defines several helper functions for the text classification tasks.

<b><i>dataset_length</i></b>
- Takes a filename and a type (train, validation, or test) as inputs and prints the number of lines in the file.
- Used for logging purposes

<b><i>parse_dataset</i></b>
- Takes a filename and a prediction class (fine or coarse) as inputs and parses the file to return a list of questions, coarse or fine classes. 
- Also returns the maximum sentence length in the dataset to be later used to pad the sentences that fall short than this length.

<b><i>batch_prediction</i></b>
- Takes a validation or testing dataset, along with other arguments including the model, the labels and mode (validation or testing) as inputs
- Sets the model to evaluation mode and uses it to predict the class of each question in the supplied list
- Prints the predicted class, true class, and sentence for each question
- Also calculates and prints the F1 score during either testing or validation.

>**Note:** The code also sets the random seed for reproducibility and checks if a GPU is available to use.




