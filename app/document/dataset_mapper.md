# DatasetMaper

<b>DatasetMaper</b> is a custom dataset using the PyTorch <b>Dataset</b> class. It takes two arguments x and y which represent the input data and target labels, respectively.

- The <i>__init__</i> method initializes the dataset by storing the input data and target labels as instance variables.
- The <i>__len__</i> method returns the length of the dataset, which is equal to the length of the input data.
- The <i>__getitem__</i> method returns a single sample from the dataset at a given index idx. It does this by returning a tuple of the input data at index <i>idx</i> and the corresponding target label at the same index.

An instance of DatasetMaper can be used with PyTorch's data loader allowing for efficient loading of data in batches during model training.