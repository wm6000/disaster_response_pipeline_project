#processdata begin

# import libraries
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
stop = stopwords.words('english')
from nltk.stem.wordnet import WordNetLemmatizer





import re

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# load data from database

df = pd.read_csv('data/disaster_messages.csv')
print(df)
X = df['message']
#Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

X = X.str.lower()
#print(X)

pattern = r"[^a-zA-Z0-9]"
X = X.str.replace(pattern, ' ')

X = X.apply(word_tokenize)

# Define a lambda function to remove stop words from each element of the series
X = X.apply(lambda x: [item for item in x if item not in stop])



print(X)

#X = X.apply(lambda x: [pos_tag(x)])

#print(X)

# Reduce words to their root form
lemmatizer = WordNetLemmatizer()
lemmed = X.apply(lambda x: [lemmatizer.lemmatize(w, pos=get_wordnet_pos(w)) for w in x])

print(X)





def tokenize(X):
    
    return X