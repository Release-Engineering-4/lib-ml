from remla_preprocess.pre_processing import MLPreprocessor
import re

def preprocess_url(url):
    """
    Preprocess a URL by removing special characters and splitting into tokens.
    """
    # Remove special characters and split into tokens
    tokens = re.findall(r'\w+', url)
    return ' '.join(tokens)

# Example usage:
url = "https://gotham-magazine.com/lalique-unveils-epis-ring"
preprocessed_url = preprocess_url(url)
print(preprocessed_url)

x = MLPreprocessor(200, None, None)

print(x.tokenizer.texts_to_sequences(["On the outbreak of the Second World War, more units were placed under Brownell's purview at RAAF Base Pearce and he was consequently promoted to temporary group captain in December"]))