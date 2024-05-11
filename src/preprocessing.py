import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import pad_sequences

OUTPUT_DIR = "google_drive_directory_path"

def save_data(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def tokenize_sequences(train_data, validation_data, test_data, sequence_length):

    training_df = train_data[["label", "url"]]
    validation_df = validation_data[["label", "url"]]
    testing_df = test_data[["label", "url"]]

    raw_x_train, raw_y_train = training_df["url"].values, training_df["label"].values
    raw_x_test, raw_y_test = testing_df["url"].values, testing_df["label"].values
    raw_x_val, raw_y_val = validation_df["url"].values, validation_df["label"].values

    tokenizer = Tokenizer(lower=True, char_level=True, oov_token="-n-")
    tokenizer.fit_on_texts(raw_x_train.tolist() + raw_x_val.tolist() + raw_x_test.tolist())

    char_index = tokenizer.word_index

    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=sequence_length)

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    tokenized_data = {
        "char_index": char_index,
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    save_data(tokenized_data, OUTPUT_DIR + "tokenized_data.pkl")