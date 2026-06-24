"""
Train Classifier
Project: Disaster Response Pipeline

Sample Script Execution:
> python models/train_classifier_LLM.py data/DisasterResponse.db

Arguments:
    1) Path to SQLite destination database (data/DisasterResponse.db)
"""

import json
import os
import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import re

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import nltk

'''

from nltk.corpus import wordnet
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.base import BaseEstimator, TransformerMixin

from tokenizer import tokenize, StartingVerbExtractor
'''

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

############################################################################################################################
'''
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
'''
############################################################################################################################    

    
def load_data(database_filepath):
    """
    Loads data from a SQLite database and prepares it for training or testing a model.

    Parameters:
    - database_filepath (str): Path to the SQLite database file.

    Returns:
    - X (pandas.Series): Input features (messages).
    - Y (pandas.DataFrame): Target variables (categories).
    - category_names (list): List of category names.
    """
    
    # Create an engine that connects to the database file
    engine = create_engine('sqlite:///' + database_filepath)

    # Load the table into a Pandas DataFrame
    df = pd.read_sql(database_filepath, engine)
    
    # Sample first 5000 rows
    # Allows for quicker processing while debugging
    #df = df.sample(n=5000)
    
    # Extract input features (messages)
    X = df['message']
    
    # Extract target variables (categories)
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    # Get the category names
    category_names = df.columns[4:].tolist()
    
    return X, Y, category_names




############################################################################

def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Evaluate the performance of the model on the test data and display the results.

    Args:
        model (object): Trained model object to evaluate.
        X_test (array-like): Test data features.
        Y_test (array-like): True labels for the test data.
        category_names (list): List of category names.

    Returns:
        tuple: A tuple containing average accuracy, average positive accuracy, and average negative accuracy.
    """
    
    # Predict on test data
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=Y_test.columns, index=Y_test.index)

    ave_accuracy1 = 0
    ave_positive_accuracy1 = 0
    ave_negative_accuracy1 = 0

    # Iterate over each category
    for cat in category_names:
        print('###################' ,cat, '###################')
        # Display results for the current category
        ave_accuracy, ave_positive_accuracy, ave_negative_accuracy = display_results(Y_test[cat], Y_pred[cat])

        # Accumulate average accuracy metrics
        ave_accuracy1 += ave_accuracy
        ave_positive_accuracy1 += ave_positive_accuracy
        ave_negative_accuracy1 += ave_negative_accuracy

    # Calculate average accuracy metrics
    ave_accuracy1 /= len(category_names)
    ave_positive_accuracy1 /= len(category_names)
    ave_negative_accuracy1 /= len(category_names)


    print('###############################################################################################')
    # Print average accuracy metrics
    print("Average Accuracy:", ave_accuracy1)
    print("Average Positive Accuracy:", ave_positive_accuracy1)
    print("Average Negative Accuracy:", ave_negative_accuracy1)

    # Print classification report
    print(classification_report(Y_test, Y_pred, target_names=category_names, zero_division=1.0))

    
def display_results(y_test, y_pred):
    
    """
    Display the confusion matrix and performance metrics based on the predicted and true labels.

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        tuple: A tuple containing accuracy, positive accuracy, and negative accuracy.
    """
    
    # Create an array of labels for the confusion matrix
    labels = np.array([0, 1])
    
    # Compute the confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Compute the overall accuracy
    accuracy = (y_pred == y_test).mean()
    
    # Extract the true positive, false negative, true negative, and false positive values
    true_positive = confusion_mat[1][1]
    false_negative = confusion_mat[1][0]
    true_negative = confusion_mat[0][0]
    false_positive = confusion_mat[0][1]
    
    # Compute the positive accuracy
    if true_positive == 0 and false_negative == 0:
        # If there are no positive samples, set positive accuracy to 1
        positive_accuracy = 1
    elif true_positive == 0:
        # If there are no true positive predictions, set positive accuracy to 0
        positive_accuracy = 0
    else:
        # Compute positive accuracy as the ratio of true positives to the sum of true positives and false negatives
        positive_accuracy = true_positive / (true_positive + false_negative)
    
    # Compute the negative accuracy
    if true_negative == 0 and false_positive == 0:
        # If there are no negative samples, set negative accuracy to 1
        negative_accuracy = 1
    elif true_negative == 0:
        # If there are no true negative predictions, set negative accuracy to 0
        negative_accuracy = 0
    else:
        # Compute negative accuracy as the ratio of true negatives to the sum of true negatives and false positives
        negative_accuracy = true_negative / (true_negative + false_positive)

    # Print the confusion matrix and performance metrics
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("Positive Accuracy:", positive_accuracy)
    print("Negative Accuracy:", negative_accuracy)
    
    # Return the computed performance metrics
    return accuracy, positive_accuracy, negative_accuracy

class OpenAIDisasterClassifier:
    """
    OpenAI-backed classifier with a sklearn-like fit/predict interface.
    """

    def __init__(self, category_names, model=None, batch_size=10):
        if OpenAI is None:
            raise ImportError(
                "The openai package is required. Install it with: pip install openai"
            )

        self.category_names = list(category_names)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.batch_size = int(os.getenv("OPENAI_BATCH_SIZE", batch_size))

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is not set. Set it before running this script."
            )

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def fit(self, X, y=None):
        """
        Keep the same call pattern as sklearn models. The OpenAI model is not
        trained locally, but category names can be refreshed from y.
        """
        if y is not None and hasattr(y, "columns"):
            self.category_names = list(y.columns)
        return self

    def predict(self, X):
        """
        Predict one binary label per disaster category for every message in X.
        """
        messages = list(X)
        predictions = []

        for start in range(0, len(messages), self.batch_size):
            batch = messages[start:start + self.batch_size]
            print(
                "Classifying messages {}-{} of {} with OpenAI...".format(
                    start + 1, start + len(batch), len(messages)
                )
            )
            predictions.extend(self._predict_batch(batch))

        return np.array(predictions, dtype=int)

    def _predict_batch(self, messages):
        category_list = ", ".join(self.category_names)
        numbered_messages = [
            {"id": index, "message": message}
            for index, message in enumerate(messages)
        ]

        system_prompt = (
            "You classify disaster-response messages. These messages come from real-world "
            "disasters, including tweets, direct messages, and news articles from events "
            "such as the 2010 Haiti earthquake, 2010 Chile earthquake, 2010 Pakistan floods, "
            "and 2012 super-storm Sandy. "
            "For each message, return a binary 0 or 1 label for every category so the "
            "information can be routed to the right response organization or team. "
            "Use 1 only when the message clearly belongs to that category. "
            "Return valid JSON only."
        )
        user_prompt = (
            "Categories: {categories}\n\n"
            "Messages:\n{messages}\n\n"
            "Return exactly this JSON shape:\n"
            '{{"predictions":[{{"id":0,"labels":{{"category_name":0}}}}]}}\n'
            "Each labels object must include every category exactly once."
        ).format(
            categories=category_list,
            messages=json.dumps(numbered_messages, ensure_ascii=True)
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        content = response.choices[0].message.content
        try:
            data = json.loads(content)
            prediction_rows = data["predictions"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("OpenAI response was not valid prediction JSON.") from exc

        rows_by_id = {int(row["id"]): row.get("labels", {}) for row in prediction_rows}
        return [
            self._labels_to_row(rows_by_id.get(index, {}))
            for index in range(len(messages))
        ]

    def _labels_to_row(self, labels):
        return [
            1 if labels.get(category) in [1, "1", True, "true", "True"] else 0
            for category in self.category_names
        ]


def build_model(category_names):
    """
    Build an OpenAI API classifier with the same fit/predict interface used by
    evaluate_model().
    """
    return OpenAIDisasterClassifier(category_names)

############################################################################

def main():
    
    """Train Classifier Main function
    
        This function applies the OpenAI classification pipeline:
        1) Extract data from SQLite db
        2) Create an OpenAI-backed classifier
        3) Estimate model performance on test set"""
       
    
    if len(sys.argv) in [2, 3]:
        
        #progrmatically pass filepaths if desired
        #database_filepath = 'data/DisasterResponse.db' 
        
        database_filepath = sys.argv[1]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        # Load data from the specified database file
        X, Y, category_names = load_data(database_filepath)

        # Split the data into training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        # Build the model
        model = build_model(category_names)

        print('Preparing OpenAI classifier...')
        # Match the sklearn call pattern; no local training happens here.
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        # Evaluate the trained model on the test set
        evaluate_model(model, X_test, Y_test, category_names)

        print('OpenAI evaluation complete. No model file was saved.')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument. \n\nExample: python '\
              'train_classifier_LLM.py ../data/DisasterResponse.db')


if __name__ == '__main__':
    main()
