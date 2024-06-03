# lib-ml

This Python library is designed for preprocessing text data in machine learning. It provides functions for tokenizing data, padding sequences, and encoding labels, all essential for training ML models. Additionally, it facilitates storing and loading data in various formats from disk. The library is accessible on PyPI and can be seamlessly integrated into your projects.

## Features

- **Data Tokenization:** Convert text into sequences of integers.
- **Sequence Padding:** Pad sequences to a consistent fixed length. 
- **Label Encoding:** Convert labels into numerical format.
- **Data Storage:** Store data to given path under selected format.
- **Data Loading:** Load data from disk under selected format.    

# Installation 

Install the library (latest version) from PyPI using: 

```bash
pip install remla-preprocess 
```

## Usage 

Example of how to use `lib-ml` for text processing: 

```python
from remla_preprocess.pre_processing import MLPreprocessor

# Instantiate the MLPreprocessor class
preprocessor = MLPreprocessor()

# Now you can use the functions of the MLPreprocessor class
preprocessor.tokenize_pad_encode_data(train_data, validation_data, test_data)
```

## Testing 

To run the tests for the pre-processing library use:

```bash
pytest
```

To run the tests with coverage for the pre-processing library use:

```bash
coverage run -m pytest
```

To generate the coverage report use:

```bash
coverage report -m
```

To generate the html of the coverage report use:

```bash
coverage html
```

## Support 
If you encounter any problems or bugs with `lib-ml`, feel free to open an issue on the project repository.