import pandas as pd
import h5py
import numpy as np
nltk.download("stopwords")
nltk.download("wordnet")
import nltk
from nltk.corpus import stopwords
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
import warnings
import data_cleaning.py
from data_loader import *
warnings.filterwarnings('ignore')
plt.set_cmap('RdYlBu')


def pre_processing:
    train, valid = data_loader.load_train_data('data/train.csv')
    test = data_loader.load_test_data('data/test.csv','data/test_labels.csv')

    train = train.fillna('')
    valid = valid.fillna('')
    test = test.fillna('')

    """clean comment_text from the trainning set and testing set"""
    train_comment_text = train.comment_text.copy()
    valid_comment_text = valid.comment_text.copy()
    test_comment_text = test.comment_text.copy()

    "Pre process the data "
    train_text_processed = [pre_process(comment) for comment in train_comment_text]
    valid_text_processed = [pre_process(comment) for comment in valid_comment_text]
    test_text_processed = [pre_process(comment) for comment in test_comment_text]

    print('The 0th comment text in unprocessed training set:')
    print(train_comment_text.iloc[0])
    print('\n')
    print('The 0th comment text in clean training set:')
    print(train_text_processed[0])
    print('\n')
    print('The 0th comment text in unprocessed validation set:')
    print(valid_comment_text.iloc[0])
    print('\n')
    print('The 0th comment text in clean validation set:')
    print(valid_text_processed[0])
    print('\n')
    print('The 0th comment text in unprocessed test set:')
    print(test_comment_text.iloc[0])
    print('\n')
    print('The 0th comment text in clean test set:')
    print(test_text_processed[0])

    df_train = pd.DataFrame(data={"comment_text": train_text_processed})
    df_train.to_csv("data/cleaned_train.csv", sep=',',index=False)

    df_valid = pd.DataFrame(data={"comment_text": valid_text_processed})
    df_valid.to_csv("data/cleaned_valid.csv", sep=',',index=False)

    df_test = pd.DataFrame(data={"comment_text": test_text_processed})
    df_test.to_csv("data/cleaned_test.csv", sep=',',index=False)

    """##Tokenizing and embedding"""

    embeddim = 300

    # Tokenize and Pad

    # Create word_tokenizer
    word_tokenizer = Tokenizer()

    # Fit and run word_tokenizer
    word_tokenizer.fit_on_texts(train_text_processed + valid_text_processed  + test_text_processed)
    tokenized_train = word_tokenizer.texts_to_sequences(train_text_processed)
    tokenized_valid = word_tokenizer.texts_to_sequences(valid_text_processed)
    tokenized_test = word_tokenizer.texts_to_sequences(test_text_processed)
    w_index = tokenize

    # Extract variables
    vocab_size = len(w_index)
    print('Vocab size: {}'.format(vocab_size))
    longest = max(len(seq) for seq in tokenized_train)
    print("Longest comment size: {}".format(longest))
    average = np.mean([len(seq) for seq in tokenized_train])
    print("Average comment size: {}".format(average))
    stdev = np.std([len(seq) for seq in tokenized_train])
    print("Stdev of comment size: {}".format(stdev))
    max_len = int(average + stdev * 3)
    print('Max comment size: {}'.format(max_len))
    print()

    # Pad sequences
    processed_X_train = pad_sequences(tokenized_train, maxlen=max_len, padding='post', truncating='post')
    processed_X_valid = pad_sequences(tokenized_valid, maxlen=max_len, padding='post', truncating='post')
    processed_X_test = pad_sequences(tokenized_test, maxlen=max_len, padding='post', truncating='post')

    # Sample tokenization
    for sample_i, (sent, token_sent) in enumerate(zip(train_text_processed[:2], tokenized_train[:2])):
        print('Sequence {}'.format(sample_i + 1))
        print('  Input:  {}'.format(sent))
        print('  Output: {}'.format(token_sent))

    embeddim = 300

    # Get embeddings
    embedindex = {}
    f = open('wiki.en.vec', encoding="utf8")
    for line in f:
        values = line.rstrip().rsplit(' ', embeddim)
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32'embeddings_index[word] = coefs
    f.close()

    print('Found {} word vectors.'.format(len(embeddings_index)))

    # Build embedding matrix
    embedmatrix = np.zeros((le) + 1, embeddim))
    for word, i i.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
          [i] = embedding_vector

    # Save embeddings
    with h5py.File('embeddings.h5', 'w') as hf:
        hf.create_dataset("fasttext",  d)

   

   
