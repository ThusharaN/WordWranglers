# How to work on this project?

## <u>Cloning the project</u>
The project can be either downloaded as a zip file from the git repository [WordWranglers](https://github.com/ThusharaN/WordWranglers) or cloned using the following git command:
```sh
git clone https://github.com/ThusharaN/WordWranglers.git
```
The project can then be opened in Visual Studio Code or any other IDE.

## <u>Setting up the project</u>

Install all the necessary dependencies


```sh
python3 -m venv myvenv
```

```sh
pip install -r requirements.txt
```

Do a test run to check if the environment has been setup correctly

```sh
python3 app/src/test_run.py
```

>**Note:** Instructions for installing git can be found [here](https://git-scm.com/downloads). Most Mac/Linux machines will have git pre-installed.

## <u>Adding Dependencies</u>
Dependencies can added to the current virtual environemt by running the following command
```sh
pip install <package_name> && pip freeze > requirements.txt
```

>**Note:** Make sure you are insidethe virtual environment created by poetry before installing any dependencies

## <u>Making changes</u>
A pull request needs to be created in order to push any changes to code.

After cloning, create a local branch to work on.

```sh
git checkout master
git checkout -b <your_branch_name>
```

In case a change has been made to <b>hello.py</b>, run the following commands to commit the changes

```sh
git status // Gives the paths of the files that have been updated
git add <file_path>/hello.py
git commit -m <some_commit_message>
```
In case new packages were installed, add the updated <b>requirements.txt</b> to the commit. Push your branch to GitHub

```sh
git push origin <your_branch_name>
```

Raise a pull request through GitHub. If the request shows conflicts, run the following commands locally

```sh
git fetch origin
git rebase origin/master
```
Fix all the conflicts in the files; add, commit and push.
<br>

>**Note:** More git commands can be found [here](https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet)

## <u>Running the program</u>

Since the classifier script resides within the <i>src</i> folder of the repository, make sure to move into this folder before running the train and/or test commands
```sh
cd WordWranglers/app/src
```

Run the following command in the console to train the model:
```sh
python3 question_classifier.py --train --config [config_file_name]
```

and the following command to test the model:
```sh
python3 question_classifier.py --test --config [config_file_name]
```

The placeholder <i>[config_file_name]</i> can be any of the following files to train and test the model with different configurations:
- <b>bilstm_pretrained.yaml</b>: Train the test classifier with pretrained word embeddings using GloVe with the sentence representation given by BiLSTM
- <b>bilstm_random.yaml</b>: Train the test classifier with randomly generated word embeddings with the sentence representation given by BiLSTM
- <b>bow_pretrained.yaml</b>: Train the test classifier with pretrained word embeddings using GloVe with the sentence representation given by Bag-Of-Words (or BoW)
- <b>bow_random.yaml</b>: Train the test classifier with randomly generated word embeddings with the sentence representation given by Bag-Of-Words (or BoW)
- <b>ensemble.yaml</b>: Train the test classifier with pretrained word embeddings using GloVe with the sentence representation given by an ensemble learner which utilisez both LSTM and Bag-Of-Words


## <u>Debugging the program</u>

The program can be debugged using the VS Code debugger. The configurations have been added as part of the <i>.vscode/launch.json</i> file
