# pylint: disable=W0621
"""
Unit testing lib-ml
"""

from unittest.mock import patch
import pytest
import pandas as pd
from sklearn.calibration import LabelEncoder
from keras.preprocessing.text import Tokenizer
from remla_preprocess.pre_processing import MLPreprocessor


@pytest.fixture
def dummy_df():
    """
    Dummy dataframe data
    """
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


@pytest.fixture
def dummy_json():
    """
    Dummy json data
    """
    return {"key1": "value1", "key2": "value2"}


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


@pytest.fixture
def extended_data():
    """
    Extended input data example with additional label and URL
    """
    return pd.DataFrame(
        {
            "label": ["label1", "label2", "label3"],
            "url": [
                "http://example1.com",
                "http://example2.com",
                "http://example3.com",
            ],
        }
    )


@pytest.fixture
def metamorphic_data():
    """
    Modified input data example with a slight modification in URLs
    """
    return pd.DataFrame(
        {
            "label": ["label1", "label2"],
            "url": ["http://example1.com", "http://example2.com/mod"],
        }
    )


@pytest.fixture
def encoding_data():
    """
    Modified input data example with a slight modification in URLs
    """
    return pd.DataFrame(
        {
            "label": ["label1", "label2", "label3",
                      "label4", "label2", "label1"],
            "url": [
                "http://example1.com",
                "http://example2.com/",
                "http://example3.com/",
                "http://example4.com/",
                "http://example2.com",
                "http://example1.com/",
            ],
        }
    )


@pytest.fixture
def dummy_tokenizer_data():
    """
    Dummy tokenizer instance
    """
    return Tokenizer()


@pytest.fixture
def dummy_encoder_data():
    """
    Dummy encoder instance
    """
    return LabelEncoder()


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


def test_load_tokenizer_with_path(tmp_path, dummy_tokenizer_data):
    """
    Test custom tokenizer loading
    """
    tokenizer_path = tmp_path / "tokenizer.pkl"

    MLPreprocessor.save_pkl(
        dummy_tokenizer_data,
        tokenizer_path,
    )

    with patch.object(
        MLPreprocessor, "load_pkl", return_value=dummy_tokenizer_data
    ) as mock_load_pkl:
        preprocessor = MLPreprocessor(tok_path=tokenizer_path)
        mock_load_pkl.assert_called_once_with(tokenizer_path)
        assert (preprocessor.tokenizer.get_config()
                == dummy_tokenizer_data.get_config())


def test_load_encoder_with_path(tmp_path, dummy_encoder_data):
    """
    Test custom encoder loading
    """
    encoder_path = tmp_path / "encoder.pkl"

    MLPreprocessor.save_pkl(
        dummy_encoder_data,
        encoder_path,
    )

    with patch.object(
        MLPreprocessor, "load_pkl", return_value=dummy_encoder_data
    ) as mock_load_enc:
        preprocessor = MLPreprocessor(enc_path=encoder_path)
        mock_load_enc.assert_called_once_with(encoder_path)
        assert preprocessor.encoder == dummy_encoder_data


def test_save_load_pkl(tmp_path, dummy_json):
    """
    Test saving/loading data as pkl
    """
    file_path = tmp_path / "test.pkl"
    MLPreprocessor.save_pkl(dummy_json, file_path)
    assert file_path.exists()

    loaded_data = MLPreprocessor.load_pkl(file_path)
    assert loaded_data == dummy_json


def test_save_load_csv(tmp_path, dummy_df):
    """
    Test saving/loading data as csv
    """
    file_path = tmp_path / "test.csv"
    MLPreprocessor.save_csv(dummy_df, file_path)
    assert file_path.exists()

    loaded_df = MLPreprocessor.load_csv(file_path)
    pd.testing.assert_frame_equal(loaded_df, dummy_df)


def test_save_load_json(tmp_path, dummy_json):
    """
    Test saving/loading data as json
    """
    file_path = tmp_path / "test.json"
    MLPreprocessor.save_json(dummy_json, file_path, indent=4)
    assert file_path.exists()

    loaded_json = MLPreprocessor.load_json(file_path)
    assert loaded_json == dummy_json


def test_load_txt(tmp_path):
    """
    Test saving/loading data as text
    """
    txt_data = "Sample text data"
    txt_file = tmp_path / "test.txt"
    with open(txt_file, "w", encoding="utf-8") as file:
        file.write(txt_data)

    loaded_data = MLPreprocessor.load_txt(txt_file)
    assert loaded_data == txt_data


def test_add_new_label(preprocessed_data, extended_data):
    """
    Test adding a new label
    """
    processor = MLPreprocessor()

    processed_original = processor.tokenize_pad_encode_data(
        preprocessed_data, preprocessed_data, preprocessed_data
    )

    processed_extended = processor.tokenize_pad_encode_data(
        extended_data, extended_data, extended_data
    )

    assert len(processed_extended["label_train"]) == 3
    assert max(processed_original["label_train"]) < max(
        processed_extended["label_train"]
    )


def test_modify_url(preprocessed_data, metamorphic_data):
    """
    Test modifying a URL
    """
    processor = MLPreprocessor()

    processed_original = processor.tokenize_pad_encode_data(
        preprocessed_data, preprocessed_data, preprocessed_data
    )

    processed_modified = processor.tokenize_pad_encode_data(
        metamorphic_data, metamorphic_data, metamorphic_data
    )

    assert all(processed_original["label_train"]
               == processed_modified["label_train"])

    assert not all(
        (processed_original["url_train"]
         == processed_modified["url_train"]).flatten()
    )


def test_consistent_encoding_across_splits(encoding_data):
    """
    Test that encoding is consistent across different data splits
    """
    processor = MLPreprocessor()

    train_data = encoding_data.iloc[:4]
    test_data = encoding_data.iloc[4:]

    processed_data = processor.tokenize_pad_encode_data(
        train_data, train_data, test_data
    )

    assert set(processed_data["label_test"]).issubset(
        set(processed_data["label_train"])
    )


def test_permutation_consistency(extended_data):
    """
    Test that permuting the order of samples results in consistent encoding
    """
    processor = MLPreprocessor()

    shuffled_data = extended_data.sample(frac=1).reset_index(drop=True)
    print(shuffled_data)

    print(extended_data)

    processed_original = processor.tokenize_pad_encode_data(
        extended_data, extended_data, extended_data
    )

    processed_shuffled = processor.tokenize_pad_encode_data(
        shuffled_data, shuffled_data, shuffled_data
    )
    for url in processed_original["url_train"]:
        assert url in processed_shuffled["url_train"]


def test_data_augmentation_consistency(encoding_data):
    """
    Test that augmenting data with duplicates results in consistent encoding
    """
    processor = MLPreprocessor()

    augmented_data = pd.concat([encoding_data, encoding_data])

    processed_data = processor.tokenize_pad_encode_data(
        augmented_data, augmented_data, augmented_data
    )

    assert len(processed_data["url_train"]) == 2 * len(encoding_data)
    assert len(processed_data["label_train"]) == 2 * len(encoding_data)
