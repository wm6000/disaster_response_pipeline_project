"""
Run Website
Project: Disaster Response Pipeline

Sample Script Execution:
> python app/run.py

Arguments:
    1) Path to SQLite destination database (data/DisasterResponse.db)
    2) Path to pickle file name where ML model is saved (data/classifier.pkl)

"""


import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import re


stop = stopwords.words('english')

app = Flask(__name__)

####################################################################################

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


####################################################################################

# load data
# Create an engine that connects to the database file
engine = create_engine('sqlite:///' + 'data/DisasterResponse.db')

# Load the table into a Pandas DataFrame
df = pd.read_sql('data/DisasterResponse.db', engine)

# load model
model = joblib.load('models/classifier.pkl')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    

    # extract data needed for visuals
    category_names = df.iloc[:,4:].columns
    category_totals = (df.iloc[:,4:] != 0).sum().values
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graphs  = [
            # GRAPH 1 - genre graph
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_totals
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }
            }
        },
            # GRAPH 2 - category graph    
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    #run website
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()