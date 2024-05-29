# pylint: disable=W0621
"""
Unit testing lib-ml
"""

import pytest
import pandas as pd
from remla_preprocess.pre_processing import MLPreprocessor


@pytest.fixture
def raw_data():
    """
    Raw input example
    """
    content = "label1\thttp://example1.com\nlabel2\thttp://example2.com"
    return content


@pytest.fixture
def preprocessed_data():
    """
    Pre-processed input example
    """
    data = {
        "label": ["label1", "label2"],
        "url": ["http://example1.com", "http://example2.com"],
    }
    return pd.DataFrame(data)


def test_split_data_content(raw_data):
    """
    Test data splitting
    """
    processor = MLPreprocessor()
    df = processor.split_data_content(raw_data)
    assert not df.empty
    assert list(df.columns) == ["label", "url"]
    assert len(df) == 2


def test_tokenize_pad_data(preprocessed_data):
    """
    Test data padding and tokenization
    """
    processor = MLPreprocessor()
    processor.tokenizer.fit_on_texts(preprocessed_data["url"])
    padded_data = processor.tokenize_pad_data(preprocessed_data["url"])
    assert padded_data.shape[0] == 2
    assert padded_data.shape[1] == processor.sequence_length


def test_tokenize_pad_encode_data(preprocessed_data):
    """
    Test data encoding
    """
    processor = MLPreprocessor()
    processed_data = processor.tokenize_pad_encode_data(
        preprocessed_data, preprocessed_data, preprocessed_data
    )
    assert "tokenizer" in processed_data
    assert "url_train" in processed_data
    assert "label_train" in processed_data
    assert len(processed_data["url_train"]) == 2
    assert len(processed_data["label_train"]) == 2
