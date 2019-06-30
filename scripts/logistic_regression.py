from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from scipy.sparse import hstack
from sklearn.pipeline import make_union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pre_processing
import pandas as pd
import data_loader.py
import numpy as np


train, valid = data_loader.load_train_data('data/train.csv', valid_rate=0.1)
train = train.fillna('')
valid = valid.fillna('')
test = data_loader.load_test_data('data/test.csv','data/test_labels.csv').fillna('')

import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def word_tokenize(s): return re_tok.sub(r' \1 ', s).split()


replacement_dict = {
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " frown ",
    ":(": " frown ",
    ":s": " frown ",
    ":-s": " frown ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}

train_data_new = []
test_data_new = []
valid_data_new = []

list_train = train['comment_text'].tolist()
list_test = test['comment_text'].tolist()
list_valid = valid['comment_text'].tolist()

for i in list_train:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in replacement_dict.keys():
            j = replacement_dict[j]
        xx = xx + j + " "
    new_train_data.append(xx)

for i in list_new_test:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in replacement_dict.keys():
            j = replacement_dict[j]
        xx = xx + j + " ".append(x_newx)
    
for i in list_valid:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in replacement_dict.keys():
            j = replacement_dict[j]
        xx = xx + j + " ".append(xx_new)

train["clean_comment_text"] = train_data_new
test["clean_comment_text"] = test_data_new
valid["clean_comment_text"] = valid_data_new

pattern = re.compile(r'[^a-zA-Z ?!]+')
train_text = train["clean_comment_text"].tolist()
test_text = test["clean_comment_text"].tolist()
valid_text = valid["clean_comment_text"].tolist()
for i,c in enumerate(train_text):
    train_text[i] = pattern.sub('',train_text[i].lower())
for i,c in enumerate(test_text):
    test_text[i] = pattern.sub('',test_text[i].lower())
for i,c in enumerate(valid_text):
    valid_text[i] = pattern.sub('',valid_text[i].lower())


train['comment_text'] = train_text
test["comment_text"] = test_text
valid["comment_text"] = valid_text
del train_text, test_text, valid_text
train.drop(['clean_comment_text'], inplace = True, axis = 1)
test.drop(['clean_comment_text'], inplace = True, axis = 1)
valid.drop(['clean_comment_text'], inplace = True, axis = 1)

all_text = pd.concat([train['comment_text'],valid['comment_text'], test['comment_text']])

word_vectorizer = TfidfVectorizer(ngram_range =(1,3),
                             tokenizer=word_tokenize,
                             min_df=3, max_df=0.9,
                             strip_accents='unicode',
                             stop_words = 'english',
                             analyzer = 'word',
                             use_idf=1,
                             smooth_idf=1,
                             sublinear_tf=1 )

char_vectorizer = TfidfVectorizer(ngram_range =(1,4),
                                 min_df=3, max_df=0.9,
                                 strip_accents='unicode',
                                 analyzer = 'char',
                                 stop_words = 'english',
                                 use_idf=1,
                                 smooth_idf=1,
                                 sublinear_tf=1,
                                 max_features=50000)

vectorizer = make_union(word_vectorizer, char_vectorizer)

vectorizer.fit(all_text)

train_matrix =vectorizer.transform(train['comment_text'])
test_matrix = vectorizer.transform(test['comment_text'])
valid_matrix = vectorizer.transform(valid['comment_text'])

test_score = []
val_score = []
def scoring_model(model, cl):
      model.fit(train_matrix, train[cl])
      pred_valid = model.predict(valid_matrix)
      pred_test = model.predict(test_matrix)
      score_valid = roc_auc_score(valid[cl], pred_valid)
      score_test = roc_auc_score(test[cl], pred_test)
      val_score.append(score_valid.mean())
      test_score.append(score_test.mean())
      print(cl)
      print(score_valid)
      print(score_test)

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
class_names = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

model = MultinomialNB()
for cl in class_names:
    scoring_model(model, cl)

MNB_score_val = val_score
MNB_score_test = test_score
print(MNB_score_val)
print(MNB_score_test)

from sklearn.linear_model import LogisticRegression
val_score = []
test_score = []
LR_model = LogisticRegression(C=3, dual=True)
for cl in class_names:
    scoring_model(LR_model,cl)

LR_score_val = val_score
LR_score_test = test_score
print(LR_score_val)
print(LR_score_test)

def pr(y_i, y):
    p = train_matrix[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=3, dual=True)
    x_nb = train_matrix.multiply(r)
    return m.fit(x_nb, y), r

model = LogisticRegression(C=3,dual = True)
NBLR_score_val=[]
NBLR_score_test=[]
for cl in class_names:
    y = train[cl].values
    r = np.log(pr(1,y) / pr(0,y))
    x_nb = train_matrix.multiply(r)
    model.fit(x_nb, y)
    pred_valid = model.predict(valid_matrix)
    pred_test = model.predict(test_matrix)
    score_valid = roc_auc_score(valid[cl], pred_valid)
    score_test = roc_auc_score(test[cl], pred_test)
    print(cl)
    print(score_valid)
    print(score_test)
    NBLR_score_val.append(score_valid.mean())
    NBLR_score_test.append(score_test.mean())


DF_score = pd.DataFrame(index=class_names)
DF_score['MNB-valid'] = MNB_score_val
DF_score['MNB-test'] = MNB_score_test
DF_score['LR-valid'] = LR_score_val
DF_score['LR-test'] = LR_score_test
DF_score['NBLR-valid'] = NBLR_score_val
DF_score['NBLR-test'] = NBLR_score_test
print(DF_score)

print("MNB validiation AUC ROC", sum(DF_score["MNB-valid"])/6)
print("MNB test AUC ROC", sum(DF_score["MNB-test"])/6)
print("LR validiation AUC ROC", sum(DF_score["LR-valid"])/6)
print("LR test AUC ROC", sum(DF_score["LR-test"])/6)
print("NBLR validiation AUC ROC", sum(DF_score["NBLR-valid"])/6)
print("NBLR test AUC ROC", sum(DF_score["NBLR-test"])/6)

preds = np.zeros((len(test), len(class_names)))

for i, j in enumerate(class_names):
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_matrix.multiply(r))[:,1]
    
np.save("nblr-svm/test_predict.npy", preds)