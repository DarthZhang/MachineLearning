import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import load_data
import config
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
import math
from keras.preprocessing.sequence import pad_sequences
import os
from keras.layers import Embedding
from keras.wrappers.scikit_learn import KerasClassifier

# fix random seed for reproducibility
np.random.seed(123)

oversample = False
full_or_partial_label = load_data.full_or_partial_label # True = full, False = Partial

# wget https://raw.githubusercontent.com/fchollet/keras/master/keras/metrics.py
# sudo cp metrics.py /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/keras/

# load the dataset but only keep the top n words, zero the rest
top_words = 20000 #this will be applied later to remove all numbers/ids over 5000 or to leave top 5000 most frequent words.

'''
Algorithm: create word index (dictionary called "labels_index"). Leave only 20 000 most frequent words. Fix length to 2000. 
Prepare embedding matrix 
'''



'''
X_train: array([list([1, 14, 22, ... , 530]), list([1, 194, 1153, ... , 2])] 
Each list is of different length. 
y_train: array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0])
'''

# Load data
##============================================================================================
import sys
import importlib
importlib.reload(config)
path = config.corpus

# sys.path.append(path)
#
# def oversample(Xtrain,Ytrain,label):
#     # Separate majority and minority classes
#     df_labels = pd.Series(np.array(Ytrain))
#     df_posts = pd.Series(np.array(Xtrain))
#     df = pd.concat([df_labels,df_posts], axis=1)
#
#     if label == 'type_1':
#         df_majority = df[df.iloc[:,0]  == 'I']
#         df_minority = df[df.iloc[:,0]  == 'E']
#     elif label == 'type_2':
#         df_majority = df[df.iloc[:,0]  == 'N']
#         df_minority = df[df.iloc[:,0]  == 'S']
#     elif label == 'type_3':
#         df_majority = df[df.iloc[:,0]  == 'F']
#         df_minority = df[df.iloc[:,0]  == 'T']
#     elif label == 'type_4':
#         df_majority = df[df.iloc[:,0]  == 'P']
#         df_minority = df[df.iloc[:,0] == 'J']
#     # Upsample minority class
#     df_minority_upsampled = resample(df_minority,
#                                      replace=True,  # sample with replacement
#                                      n_samples=df_majority.shape[0],  # to match majority class
#                                      random_state=123)  # reproducible results
#     # Combine majority class with upsampled minority class
#     df_upsampled = pd.concat([df_majority, df_minority_upsampled])
#     return df_upsampled
#

## Load dataset
# =====================================================================================
df = load_data.df

X1 = load_data.X1
Xtrain = load_data.Xtrain
Xtest = load_data.Xtest

if full_or_partial_label:
    Ytrain = load_data.Ytrain
    Ytest = load_data.Ytest

#one hot encode
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Ytrain)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Ytrain_integer = [item for sublist in integer_encoded for item in sublist]
Ytrain_encoded = onehot_encoder.fit_transform(integer_encoded)

# invert first example
# inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
# print(inverted)





# preprocess, one_hot_encode
# ==================================================================================================
'''
this could have all been done with to_categorical(), and sequence.pad_sentences. 
'''
# from keras.preprocessing.text import one_hot
#
# one_hot_encode and zero-pad
# def one_hot_encode(X):
#     X1_encoded = []
#     for text in X:
#         words = set(text_to_word_sequence(text)) # estimate the size of the vocabulary
#         vocab_size = len(words)
#         result = one_hot(text, round(vocab_size*1.3)) # integer encode the document/One_hot_encode/hash See: https://www.quora.com/Can-you-explain-feature-hashing-in-an-easily-understandable-way
#         # truncate and pad input sequences
#         result = sequence.pad_sequences(np.array([result]), maxlen=max_review_length)  # zero pads at the beggining of list: array([[    0,     0,     0, ..., 1, 14], [    0,     0,     0, ..., 145,    95]], dtype=int32)
#         X1_enc.append(result[0])
#     X1_encoded = np.array(X1_enc)
#     return X1_enc
#
# Xtrain_encoded = one_hot_encode(Xtrain)
# Xtest_encoded = one_hot_encode(Xtest)

# ==================================================================================================

tokenizer = Tokenizer(num_words=20000) #TODO: change for my project, use all. This basically removes all the URLs and random tokens ''4plebs', 82886), ('1371', 82887), ('1371531897133', 82888), ('13z4l8jvbpy', 82889), ('deebkni', 82890)
tokenizer.fit_on_texts(Xtrain)
sequences = tokenizer.texts_to_sequences(Xtrain)
word_index = tokenizer.word_index
word_index.items()
print('Found %s unique tokens.' % len(word_index))

#define max length of doc
l1 = []
for text in X1:
    l1.append(len(text_to_word_sequence(text)))

def roundup(x):
    return int(math.ceil(x / 100.0)) * 100

MAX_SEQUENCE_LENGTH = roundup(np.max(l1))
Xtrain_encoded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
sequences2 = tokenizer.texts_to_sequences(Xtest)
Xtest_encoded = pad_sequences(sequences2, maxlen=MAX_SEQUENCE_LENGTH)
## embedding layer https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# =====================================================================================
importlib.reload(config)

embeddings_index = {}
# with open(os.path.join(config.word2vec_path,'GoogleNews-vectors-negative300.bin')) as f:
with open(os.path.join(config.word2vec_path,'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# leverage our embedding_index dictionary and our word_index to compute our embedding matrix
# ============================================================================================================
EMBEDDING_DIM = 100
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load embedding matrix into a embedding layer
# ============================================================================================================
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=2000,
                            trainable=False)

#Create model, run model on different Y values
# ============================================================================================================
def create_model(dropout_rate=0.0):
    model = Sequential()
    model.add(Embedding(20000, 100, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(100))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

np.random.seed(123)

model = KerasClassifier(build_fn=create_model, verbose=1, batch_size=16,epochs=10)

#tune different hyperparameters:
# param_grid0 = dict(epochs=[2],batch_size = [8,16,32,64])
# param_grid1 = dict(optimizer = ['SGD', 'Adam', 'Adamax', 'Nadam']) #add epochs and batch_size to Keras Classfier
# param_grid2 = dict(activation = ['linear', 'relu', 'sigmoid']) # also change function to activation='relu'
param_grid3 = dict(dropout_rate = [0.0,0.3,0.5]) # add parameters to function dropout_rate=0.0, weight_constraint=0

grid = GridSearchCV(estimator=model, param_grid=param_grid3, n_jobs=1, scoring='f1_weighted', verbose=1)
grid_result = grid.fit(Xtrain_encoded, Ytrain_integer)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#final model
model = Sequential()
model.add(Embedding(20000, 100, input_length=MAX_SEQUENCE_LENGTH))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(Xtrain_encoded, Ytrain_encoded, validation_split=0.2, epochs=20, batch_size=16, verbose=1)
loss, accuracy = model.evaluate(Xtest_encoded, Ytest, verbose=1)
print('Accuracy: %f' % (accuracy * 100))