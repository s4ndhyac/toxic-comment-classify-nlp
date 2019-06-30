import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import string
import pre_processing
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, GRU,Embedding, Dropout, Activation, Flatten, Bidirectional
from keras.layers import SpatialDropout1D, concatenate,Bidirectional, GRU, GlobalAveragePooling1D, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.layers.convolutional import Conv1D, MaxPooling1D

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

# Load embeddings
with h5py.File('embeddings.h5', 'r') as hf:
        embedding_matrix = hf['fasttext'][:]

"""# Model Architecture CNN"""

max_features = 20000
max_len=250

# build CNN model
model_cnn = Sequential()
model_cnn.add(Embedding(max_features, 100, input_length=max_len))
model_cnn.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Dropout(0.5))
model_cnn.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Dropout(0.5))
model_cnn.add(Flatten())
model_cnn.add(Dense(600, activation="relu"))
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(6, activation="sigmoid"))

#summary
model_cnn.summary()

"""###Train Model"""

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.9, random_state=30)

model_cnn_trained, history_cnn = train_model(model_cnn,
                                             X_train,y_train,
                                             X_val,y_val,
                                             batch_size=512, 
                                             epochs=3,
                                             filepath='models/weights.best.from_scatch_cnn.hdf5')

"""###Evaluation Plot"""

history_plot(history_cnn,'history_cnn.eps')

plot_model(model_cnn, to_file='figure/model_cnn.eps', show_shapes=True, show_layer_names=True)

"""#Transfer Learning (GloVe) model"""

# load the whole embedding into memory
def create_embeddings_index(filename):
    embeddings_index = dict()
    f = open(filename, encoding = 'utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index.keys()))
    return embeddings_index

# create a weight matrix for words in training docs (look up word vectors in embeddings_index)
def create_embeddings_matrix(tokenizer, embeddings_index):
    vocab_size = len(tokenizer.word_index) + 1
    embeddings_matrix = np.zeros((vocab_size, 100))
    for word, i in tokenizer.word_index.items():
        embeddings_vector = embeddings_index.get(word)
        if embeddings_vector is not None:
            embeddings_matrix[i] = embeddings_vector
    return embeddings_matrix
