
#### RNN with LSTM

import keras.backend
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.layers import Dropout, GlobalMaxPooling1D, BatchNormalization
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Nadam
from keras.layers.recurrent import LSTM

# Load embeddings
with h5py.File('embeddings.h5', 'r') as hf:
    embedding_matrix = hf['fasttext'][:]

# Initate model
model = Sequential()

# Add Embedding layer
model.add(Embedding(vocab_size + 1, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=True))

# Add Recurrent layer
model.add(LSTM(60, return_sequences=True, name='lstm_layer'))
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(3))
model.add(GlobalMaxPooling1D())
model.add(BatchNormalization())

# Add fully connected layers
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(6, activation='sigmoid'))

# Summarize the model
model.summary()

"""Using binary crossentropy as the loss function and clipping gradients to avoid any explosions."""

def loss(y_true, y_pred):
     return keras.backend.binary_crossentropy(y_true, y_pred)

lr = .0001
model.compile(loss=loss, optimizer=Nadam(lr=lr, clipnorm=1.0),
              metrics=['binary_accuracy'])

"""#### Training Model"""

# Evaluation Metric
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

class RocAucEvaluation(Callback):
    def __init__(self, filepath, validation_data=(), test_data=(), interval=1, max_epoch = 100):
        super(Callback, self).__init__()
        # Initialize state variables
        print("After init")
        self.interval = interval
        self.filepath = filepath
        self.stopped_epoch = max_epoch
        self.best = 0
        self.X_val, self.y_val = validation_data
        self.y_pred = np.zeros(self.y_val.shape)
        self.X_test, self.y_test = test_data
        self.y_test_pred = np.zeros(self.y_test.shape)

    def on_epoch_end(self, epoch, logs={}):
        print("Epoch end")
        if epoch % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            current = roc_auc_score(self.y_val, y_pred)
            logs['roc_auc_val'] = current
            y_test_pred = self.model.predict_proba(self.X_test, verbose=0)
            current_test = roc_auc_score(self.y_test, y_test_pred)
            print("Test: {:.5f} Val:{:.5f}".format(current_test, current))

            if current > self.best: #save model
                print(" - AUC - improved from {:.5f} to {:.5f}".format(self.best, current))
                self.best = current
                self.y_pred = y_pred
                self.stopped_epoch = epoch+1
                self.model.save(self.filepath, overwrite=True)
            else:
                print(" - AUC - did not improve")

# Training

from keras.callbacks import EarlyStopping, ModelCheckpoint
                                            
print("Starting to train model...")
file_path = 'rnn/rnn_model_best.hdf5'
RocAuc_checkpoint = RocAucEvaluation(filepath=file_path,validation_data=(processed_X_valid, valid_y), test_data=(processed_X_test, test_y), interval=1)
early_stop = EarlyStopping(monitor="roc_auc_val", mode="max", patience=3)
callbacks_list = [RocAuc_checkpoint, early_stop] 

hist = model.fit(processed_X_train, train_y, epochs=5, batch_size=64, shuffle=False, validation_data=(processed_X_valid, valid_y), 
                 callbacks = callbacks_list, verbose=1)    
best_score = min(hist.history['val_loss'])
print("Final ACC. {}".format(best_score))

evaluation_cost = hist.history['val_loss']
evaluation_accuracy = hist.history['val_binary_accuracy']
training_cost = hist.history['loss']
training_accuracy = hist.history['binary_accuracy']

np.save("rnn/evaluation_cost.npy", evaluation_cost)
np.save("rnn/evaluation_accuracy.npy", evaluation_accuracy)
np.save("rnn/training_cost.npy", training_cost)
np.save("rnn/training_accuracy.npy", training_accuracy)

model.load_weights('rnn/rnn_model_best.hdf5')
print("Predicting results...")
test_predicts_path = "rnn/rnn_test_predicts.npy"
test_pred = model.predict(processed_X_test, batch_size=1024, verbose=1)
np.save(test_predicts_path, test_pred)

# Visualize history of loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Visualize history of accuracy
plt.plot(hist.history['binary_accuracy'])
plt.plot(hist.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('binary_accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""#### Submission"""

preds_loaded = np.load("rnn/rnn_test_predicts.npy")
predictions = pd.DataFrame(preds_loaded)
sample_submission = pd.read_csv("data/sample_submission.csv")
sample_submission[list_classes] = predictions
sample_submission.to_csv("rnn/submission.csv", index=False)

## test roc auc score
from sklearn.metrics import roc_auc_score
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
preds_loaded = np.load("rnn/rnn_test_predicts.npy")
predictions = pd.DataFrame(preds_loaded)
test = data_loader.load_test_data('data/test.csv','data/test_labels.csv')
roc_auc_scores_test = 0
for class_name in class_names:
    score = roc_auc_score(test[class_name], predictions[class_name])
    roc_auc_scores_test += score
    print(score)
print("ROC AUC Test score:", roc_auc_scores_test/6)