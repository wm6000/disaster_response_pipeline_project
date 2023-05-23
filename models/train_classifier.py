"""
Train Classifier
Project: Disaster Response Pipeline

Sample Script Execution:
> python models/train_classifier.py data/DisasterResponse.db data/classifier.pkl

Arguments:
    1) Path to SQLite destination database (data/DisasterResponse.db)
    2) Path to pickle file name where ML model is saved (data/classifier.pkl)
"""


import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import re


import pickle
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

#from tokenizefunctions import StartingVerbExtractor
#from tokenizefunctions import get_wordnet_pos
#from tokenizefunctions import tokenize



# Import tools needed for visualization
from sklearn import tree
import pydot
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from rake_nltk import Rake

############################################################################################################################

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





############################################################################

############################################################################

def build_model():
    """
    Build a machine learning model pipeline.

    Returns:
    - cv (GridSearchCV): Grid search object for model training and tuning.

    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators=50)))
    ])

    #parameters = {
    #    'clf__estimator__n_estimators': [50, 100, 200]
    #}
    
    # Perform grid search to find the best model parameters
    #cv = GridSearchCV(pipeline, param_grid=parameters)

    #return cv
    return pipeline

def save_model(model, model_filepath):
    """
    Save the trained model to a file using pickle.

    Args:
        model (object): Trained model object to be saved.
        model_filepath (str): Filepath to save the model.

    Returns:
        None
    """
    # Use pickle to dump the model object to the specified file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

############################################################################

def main():
    
        """
        Train Classifier Main function
    
        This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
        """
    
    #if len(sys.argv) == 3:
        database_filepath = 'data/DisasterResponse.db' 
        model_filepath = 'models/classifier.pkl'
        #database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        # Load data from the specified database file
        X, Y, category_names = load_data(database_filepath)

        # Split the data into training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        # Build the model
        model = build_model()

        print('Training model...')
        # Train the model using the training data
        model.fit(X_train, Y_train)


        # Get the best parameters
        #best_params = model.best_params_

        # Print the best parameters
        #print("Best parameters:")
        #for param, value in best_params.items():
        #    print(f"{param}: {value}")

        ##########################################################

        #clf = model.named_steps['clf'].estimators_[0].estimator_
        
        #plt.figure(figsize=(15,10))
        #tree.plot_tree(clf,filled=True)
        #plt.show()
        ##########################################################

        print('Evaluating model...')
        # Evaluate the trained model on the test set
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        # Save the trained model to the specified file path
        save_model(model, model_filepath)

        print('Trained model saved.')

    #else:
    #    print('Please provide the filepath of the disaster messages database '\
    #          'as the first argument and the filepath of the pickle file to '\
    #          'save the model to as the second argument. \n\nExample: python '\
    #          'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()