


import re
import pandas as pd

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer class that extracts features indicating if a sentence starts with a verb.

    Parameters:
    - None

    Methods:
    - starting_verb(text): Checks if a sentence in the given text starts with a verb.
    - fit(x, y=None): Fits the transformer to the data. Returns self.
    - transform(X): Applies the transformation to the input data X and returns the result as a DataFrame.
    """

    def starting_verb(self, text):
        """
        Checks if a sentence in the given text starts with a verb.

        Parameters:
        - text (str): Input text to check.

        Returns:
        - bool: True if a sentence starts with a verb, False otherwise.
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(verbtokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        """
        Fits the transformer to the data.

        Parameters:
        - x: Input features (ignored).
        - y: Target values (ignored).

        Returns:
        - self: The fitted transformer object.
        """
        return self

    def transform(self, X):
        """
        Applies the transformation to the input data X.

        Parameters:
        - X: Input data to transform.

        Returns:
        - pandas.DataFrame: Transformed data containing a column indicating if each sentence starts with a verb.
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    

def get_wordnet_pos(word):
    """
    Map POS tag to the first character that lemmatize() accepts.

    Parameters:
    - word (str): The word for which the POS tag needs to be mapped.

    Returns:
    - pos_tag (str): The mapped POS tag.

    """
    # Perform POS tagging on the word and extract the tag of the first word
    tag = nltk.pos_tag([word])[0][1][0].upper()

    # Define a dictionary mapping POS tags to corresponding WordNet POS tags
    tag_dict = {"J": wordnet.ADJ,  # Adjective
                "N": wordnet.NOUN,  # Noun
                "V": wordnet.VERB,  # Verb
                "R": wordnet.ADV}  # Adverb

    # Return the corresponding WordNet POS tag using the tag dictionary,
    # defaulting to Noun if the tag is not found in the dictionary
    return tag_dict.get(tag, wordnet.NOUN)


def tokenize(text):
    """
    Tokenize and preprocess the input text.

    Parameters:
    - text (str): The text to be tokenized and preprocessed.

    Returns:
    - clean_tokens (list): The list of preprocessed tokens.

    """
    # Replace all URLs with a placeholder string 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for detected_url in detected_urls:
        text = text.replace(detected_url, 'urlplaceholder')

    # Remove non-alphanumeric characters and convert text to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize the text into individual words
    tokens = word_tokenize(text)

    # Remove stopwords from the tokens
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]

    # Lemmatize the tokens using WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos=get_wordnet_pos(tok)).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens




def verbtokenize(text):
    """
    Tokenize the input text and lemmatize the verbs.

    Parameters:
    - text (str): The text to be tokenized and lemmatized.

    Returns:
    - clean_tokens (list): The list of lemmatized tokens.

    """
    # Replace all URLs with a placeholder string 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Tokenize the text into individual words
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        # Lemmatize the token to its base form (lowercase) using WordNetLemmatizer
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens