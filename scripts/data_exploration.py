import pandas as pd
import h5py
import numpy as np
from nltk.corpus import stopwords
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem.wordnet import WordNetLemmatizer
import data_loader
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline
plt.set_cmap('RdYlBu')
import pre_processing

train, valid = data_loader.load_train_data('data/train.csv')
test = data_loader.load_test_data('data/test.csv','data/test_labels.csv')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_y = train[list_classes].values
valid_y = valid[list_classes].values
test_y = test[list_classes].values

train = train.fillna('')
valid = valid.fillna('')
test = test.fillna('')

"""## Data Exploration"""

print(train.shape)
print(valid.shape)
print(test.shape)

print(train.dtypes)

print(train[0:5])

print(test[0:5])

# counting frequency of occurence of multi-labelled data
cnt0,cnt1,cnt2 = 0,0,0
label = train[['toxic', 'severe_toxic' , 'obscene' , 'threat' , 'insult' , 'identity_hate']]
label = label.as_matrix()
for i in range(label.shape[0]):
    ct = np.count_nonzero(label[i])
    if ct :
        n = n+1
    else:
        cnt0 = cnt0+1
    if ct>1 :
        n = n+1
print("Train samples with no label:", cnt0)
print("Train samples with atleast one label:", n)
print("Train samples with 2 or more labels", n)

# Explore the vocabulary
import collections
from tqdm import tqdm

x_train = train.comment_text.copy()
counter_word = collections.Counter([word for sentence in tqdm(x_train, total=len(x_train)) \
                                                              for word in sentence.split()])

print('{} words.'.format(len([word for sentence in x_train for word in sentence.split()])))
print('{} unique words.'.format(len(counter_word)))
print('10 Most common words in the dataset:')
print('"' + '" "'.join(list(zip(*counter_word.most_common(10)))[0]) + '"')

# visualizing the cmmnt size
cmmnt = train['comment_text']
cmmnt = cmmnt.as_matrix()
x = [len(cmmnt[i]) for i in range(cmmnt.shape[0])]
print('average length of cmmnt: {:.3f}'.format(sum(x)/len(x)) )
bins = [1,200,400,600,800,1000,1200]
plt.hist(x, bins=bins)
plt.xlabel('Length of comments')
plt.ylabel('Number of comments')       
plt.axis([0, 1200, 0, 90000])
plt.grid(True)
plt.show()

import seaborn as sns
# visualizing the no. of comments of each category
palette= sns.color_palette("bright")
x=train.iloc[:,2:].sum()
plt.figure(figsize=(9,6))
ax= sns.barplot(x.index, x.values, palette=palette)
plt.title("Class")
plt.ylabel('Occurrences', fontsize=12)
plt.xlabel('Type ')
rects = ax.patches
xlabels = x.values
for rect, lbl in zip(rects, xlabels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 10, lbl, 
            ha='center', va='bottom')

plt.show()

# No. of comments of each type grouped by lengths
y = np.zeros(label.shape)
for ix in range(cmmnt.shape[0]):
    l = len(cmmnt[ix])
    if label[ix][0] :
        y[ix][0] = l
    if label[ix][1] :
        y[ix][1] = l
    if label[ix][2] :
        y[ix][2] = l
    if label[ix][3] :
        y[ix][3] = l
    if label[ix][4] :
        y[ix][4] = l
    if label[ix][5] :
        y[ix][5] = l

labelsplt = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
color = ['red','green','blue','yellow','orange','chartreuse']        
plt.hist(y,bins = bins,label = labelsplt,color = color)
plt.axis([0, 1200, 0, 10000])
plt.xlabel('Length of comments')
plt.ylabel('Number of comments') 
plt.legend()
plt.grid(True)
plt.show()

# correlation matrix between features
f, ax = plt.subplots(figsize=(10, 8))
corr = train.corr()
corr.style.background_gradient()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
            square=True, ax=ax, annot=True)

