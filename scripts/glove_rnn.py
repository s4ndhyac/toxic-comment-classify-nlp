import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import string
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

# GloVe model
# load the whole embedding into memory
embeddings_index_glove = create_embeddings_index('glove.6B.100d.txt')
# create a weight matrix for the embedding layer from a loaded embedding
embeddings_matrix_glove = create_embeddings_matrix(tokenizer, embeddings_index_glove)

# define the model
vocab_size = len(tokenizer.word_index) + 1
model_glove = Sequential()
model_glove.add(Embedding(vocab_size, 100, weights = [embeddings_matrix_glove], input_length = maxlen, trainable = False))
model_glove.add(Bidirectional(LSTM(128, return_sequences=True)))
model_glove.add(GlobalMaxPool1D())
model_glove.add(Dropout(0.25))
model_glove.add(Dense(60, activation="relu"))
model_glove.add(Dropout(0.25))
model_glove.add(Dense(6, activation="sigmoid"))

# summary
model_glove.summary()

"""## Train Model"""
model_glove_trained, history_glove = train_model(model_glove,X_train,y_train,
                                     X_val,y_val,
                                     batch_size=512, 
                                     epochs=5,
                                     filepath='models/weights.best.glove.hdf5')

# more training based on the previous model training
model_glove_trained_2, history_glove_2 = train_model(model_glove_trained,
                                                     X_train,y_train,
                                                     X_val,y_val,
                                                     batch_size=512, 
                                                     epochs=5,
                                                     filepath='models/weights.best.glove_2.hdf5')

"""## Evaluation Plot"""
plot_model(model_glove, to_file='figure/model_glove.eps', show_shapes=True, show_layer_names=True)

fig = plt.figure(figsize=(15, 5))
# summarize history for accuracy
plt.subplot(1, 2, 1)
plt.plot(history_glove.history['acc']+history_glove_2.history['acc']); 
plt.plot(history_glove.history['val_acc']+history_glove_2.history['val_acc']);
plt.title('model accuracy'); plt.ylabel('accuracy');
plt.xlabel('epoch'); plt.legend(['train', 'valid'], loc='upper left');

# summarize history for loss
plt.subplot(1, 2, 2)
plt.plot(history_glove.history['loss']+history_glove_2.history['loss']); 
plt.plot(history_glove.history['val_loss']+history_glove_2.history['val_loss']);
plt.title('model loss'); plt.ylabel('loss');
plt.xlabel('epoch'); plt.legend(['train', 'valid'], loc='upper left');
plt.show()
    
fig.savefig('figure/history_glove.eps',bbox_inches = 'tight')