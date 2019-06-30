import gc
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
import lightgbm as lgb


cl_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train, valid = data_loader.load_train_data('data/train.csv')
test = data_loader.load_test_data('data/test.csv','data/test_labels.csv').fillna('')
train = train.fillna('')
valid = valid.fillna('')
test = test.fillna('')

train_text = train['comment_text']
valid_text = valid['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, valid_text, test_text])

w_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 2),
    max_features=50000)

w_vectorizer.fit(all_text)

word_train_features = w_vectorizer.transform(train_text)
word_valid_features = w_vectorizer.transform(valid_text)
word_test_features = w_vectorizer.transform(test_text)

ch_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)char_vectorizer.fit(all_text)

char_train_features char_vectorizer.transform(train_text)
char_valid_features char_vectorizer.transform(valid_text)
test_char_features char_vectorizer.transform(test_text)


train_features =char_ hsta, word_train_features])
valid_features = hstack([char_valid_features, word_valid_features])
test_features = hstack([test_char_features, word_test_features])


submission = pd.DataFrame.from_dict({'id': test['id']})

train.drop('comment_text', axis=1, inplace=True)

del train_text
del test_text
del all_char_text

del test_char_features
del word_train_features
del word_test_features
gc.collect()

for cl in cl_names:
    print(cl)
    train_target = train[cl]
    valid_target = valid[cl]
    
    model = LogisticRegression(solver='sag')
    sel_model = SelectFromModel(model, threshold=0.2)
    
    sparse_train_matrix = sel_model.fit_transform(train_features, train_target)
    sparse_valid_matrix = sel_model.fit_transform(valid_features, valid_target)
    y_train = train_target
    y_valid = valid_target
    
    sparse_test_matrix = sel_model.transform(test_features)
    
    train_dataset = lgb.Dataset(sparse_train_matrix, label=y_train)
    valid_dataset = lgb.Dataset(sparse_valid_matrix, label=y_valid)
    
    set_watchlist = [train_dataset, valid_dataset]
    params = {'learning_rate': 0.2,
              'application': 'binary',
              'num_leaves': 31,
              'verbosity': -1,
              'metric': 'auc',
              'data_random_seed': 2,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.6,
              'nthread': 4,
              'lambda_l1': 1,
              'lambda_l2': 1,
             'is_training_metric': True}
    rounds_lookup = {'toxic': 140,
                 'severe_toxic': 50,
                 'obscene': 80,
                 'threat': 80,
                 'insult': 70,
                 'identity_hate': 80}
    model = lgb.train(params,
                      train_set=train_dataset,
                      num_boost_round=rounds_lookup[cl],
                      valid_sets=set_watchlist,
                      early_stopping_rounds=5,
                      verbose_eval=10)
    submission[cl] = modesparse_l.pr)

submission.to_csv('lgb_submission.csv', index=False)



### LGBM score

from sklearn.metrics import roc_auc_score
lgbm_preds = pd.read_csv("lgbm/lgb_submission.csv")
test = data_loader.load_test_data('data/test.csv','data/test_labels.csv').fillna('')
cl_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
roc_auc_scores_test = 0
for cl in cl_names:
    score = roc_auc_score(test[cl], lgbm_preds[cl])
    roc_auc_scores_test += score
    print(score)
print("ROC AUC Test score:", roc_auc_scores_test/6)