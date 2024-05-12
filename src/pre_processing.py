# pylint: disable=E0401,R0914

"""
Data preprocessing for ML model 
"""

import gdown
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences


class MLPreprocessor:
    def __init__(self, sequence_length=200, tokenizer_path=None, encoder_path=None):
        self.sequence_length = sequence_length
        if tokenizer_path:
            self.tokenizer = MLPreprocessor.load_pkl_data(tokenizer_path)
        else:
            self.tokenizer = Tokenizer(lower=True, char_level=True, oov_token="-n-")
        if encoder_path:
            self.encoder = MLPreprocessor.load_pkl_data(encoder_path)
        else:
            self.encoder = LabelEncoder()

    def split_data_content(self, content):
        """
        Create dataframe from raw data
        """
        content_lines = content.split("\n")
        raw_data = [line.strip() for line in content_lines]
        raw_urls = [line.split("\t")[1] for line in raw_data]
        raw_labels = [line.split("\t")[0] for line in raw_data]
        df = pd.DataFrame({"label": raw_labels, "url": raw_urls})
        return df

    def tokenize_pad_data(self, data):
        """
        Tokenize and pad data sequences
        """
        return pad_sequences(
            self.tokenizer.texts_to_sequences(data), maxlen=self.sequence_length
        )

    def tokenize_pad_encode_data(self, train_data, validation_data, test_data):
        """
        Tokenize, pad, and encode data
        """
        raw_x_train, raw_y_train = (
            train_data["url"].values,
            train_data["label"].values,
        )
        raw_x_test, raw_y_test = (
            test_data["url"].values,
            test_data["label"].values,
        )
        raw_x_val, raw_y_val = (
            validation_data["url"].values,
            validation_data["label"].values,
        )

        self.tokenizer.fit_on_texts(
            raw_x_train.tolist() + raw_x_val.tolist() + raw_x_test.tolist()
        )

        x_train = self.tokenize_pad_data(raw_x_train)
        x_val = self.tokenize_pad_data(raw_x_val)
        x_test = self.tokenize_pad_data(raw_x_test)

        y_train = self.encoder.fit_transform(raw_y_train)
        y_val = self.encoder.transform(raw_y_val)
        y_test = self.encoder.transform(raw_y_test)

        return {
            "tokenizer": self.tokenizer,
            "char_index": self.tokenizer.word_index,
            "url_train": x_train,
            "url_val": x_val,
            "url_test": x_test,
            "label_train": y_train,
            "label_val": y_val,
            "label_test": y_test,
        }

    @staticmethod
    def get_data_from_gdrive(file_id, output):
        """
        Download data from Google Drive
        """
        gdown.download(file_id, output=output)

    @staticmethod
    def save_data_as_pkl(data, path):
        """
        Save data to given path into pickle format
        """
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def save_csv(df, file_path, index=False):
        """
        Save a pandas DataFrame to a CSV file for a given path
        """
        df.to_csv(file_path, index=index)

    @staticmethod
    def load_pkl_data(path):
        """
        Load pickle data from given path
        """
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def load_csv_data(path):
        """
        Load csv data from given path
        """
        data = pd.read_csv(path)
        return data

    @staticmethod
    def load_txt_data(path):
        """
        Load txt data from given path
        """
        with open(path, "r") as file:
            data = file.read()
        return data
