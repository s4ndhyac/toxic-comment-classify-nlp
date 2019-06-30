import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem.wordnet import WordNetLemmatizer

"""## Prepare the Data"""

"""this function receives comments and returns clean word-list
  split words by witespace, remove punctuations, change letters to lower case, 
  remove words that are not alphanumeric, remove 1-letter words
"""
def pre_process(word_text):
    tokens = word_text.split()
    table = str.maketrans({key: None for key in string.punctuation})
    tokens = [token.translate(table) for token in tokens]
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token.lower() for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if len(token) > 1]
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(token,"v") for token in tokens]
    sentence = ' '.join(tokens)
    return sentence