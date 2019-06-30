# Toxic Comment Classify NLP
Identify and classify toxic online comments

## Problem Description
The Toxic Comment Challenge aims to build a multi-label classification model that detects different levels of toxicity in online comments. The 6 levels being toxic, severe toxic, obscene, threat, insult and identity hate.

## Data Pre-processing and Feature Engineering
- Cleaned the data using natural language processing techniques such as removing stopwords, stemming and lemmatizing
- Converted the clean data in embeddings using TF-IDF vectorization and word2vec pre-trained model depending on the model being evaluated

## Models Evaluated
- Multinomial Naive Bayes - Logistic Regression (baseline model)
- Recurrent Neural Networks (RNNs) with Stanford's GLoVe pre-trained embedding model
- Recurrent Neural Networks with Long Short Term Memory (RNNs with LSTM) [using Facebook's fasttext pre-trained word embedding model]
- Convolutional Neural Network (CNN)
- LGBM (Light GBM) Gradient Boosting Framework

## Conclusion
The RNN with LSTM achieved the highest ROC-AUC score of 98.01 on the test dataset.
