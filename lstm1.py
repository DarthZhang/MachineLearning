# LSTM with Dropout for sequence classification in the IMDB dataset
import numpy as np
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
np.random.seed(7)

oversample = False

# wget https://raw.githubusercontent.com/fchollet/keras/master/keras/metrics.py
# sudo cp metrics.py /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/keras/



# load the dataset but only keep the top n words, zero the rest
top_words = 5000 #this will be applied later to remove all numbers/ids over 5000.
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words) # only loads in Python2
'''
X_train: array([list([1, 14, 22, ... , 530]), list([1, 194, 1153, ... , 2])] 
Each list is of different length. 
y_train: array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0])
'''

# Load data
##============================================================================================
import config
import split_types
import data_cleaner
import sys
import importlib
importlib.reload(config)
path = config.corpus
sys.path.append(path)


# Choose df
##============================================================================================
# FULL label
# df = pd.read_csv(path) #full label (e.g., INTP)
# X = df['posts'].tolist()
# Y = df['type'].tolist()

# Partial label
df = split_types.new_df  # partial label (e.g., I)


df = df.sample(frac=1, random_state=123) # randomizes whole dataframe

#Save it in order to skip loading time
# df.to_csv(config.path+'df.csv')

# X = df['posts'].tolist()

# Clean data
##============================================================
data_cleaner = data_cleaner.data_cleaner()

# X1 = data_cleaner.preprocess1(X)
# np.save(config.path+'X1',X1)


# Load pre-cleaned data, and split it
##============================================================
X1 = np.load(config.path+'X1.npy').tolist()

split_point = int(0.7 * len(X1))
Xtrain = X1[:split_point]
Xvalidation = X1[split_point:(split_point+1301)]
Xtest = X1[(split_point + 1301):-1]

# d = {}
# labels = df.columns[:-1]


def oversample(Xtrain,Ytrain,label):
    # Separate majority and minority classes
    df_labels = pd.Series(np.array(Ytrain))
    df_posts = pd.Series(np.array(Xtrain))
    df = pd.concat([df_labels,df_posts], axis=1)

    if label == 'type_1':
        df_majority = df[df.iloc[:,0]  == 'I']
        df_minority = df[df.iloc[:,0]  == 'E']
    elif label == 'type_2':
        df_majority = df[df.iloc[:,0]  == 'N']
        df_minority = df[df.iloc[:,0]  == 'S']
    elif label == 'type_3':
        df_majority = df[df.iloc[:,0]  == 'F']
        df_minority = df[df.iloc[:,0]  == 'T']
    elif label == 'type_4':
        df_majority = df[df.iloc[:,0]  == 'P']
        df_minority = df[df.iloc[:,0] == 'J']
    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=df_majority.shape[0],  # to match majority class
                                     random_state=123)  # reproducible results
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    return df_upsampled


# preprocess, one_hot_encode. To use embeddings, do this: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# ==================================================================================================
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence

#define max length of doc
l1 = []
for text in X1:
    l1.append(len(text_to_word_sequence(text)))

import math
def roundup(x):
    return int(math.ceil(x / 100.0)) * 100

max_review_length = roundup(np.max(l1))

# one_hot_encode and zero-pad, takes about 5 min
X1_enc = []
for text in X1:
    words = set(text_to_word_sequence(text)) # estimate the size of the vocabulary
    vocab_size = len(words)
    result = one_hot(text, round(vocab_size*1.3)) # integer encode the document/One_hot_encode/hash See: https://www.quora.com/Can-you-explain-feature-hashing-in-an-easily-understandable-way
    # truncate and pad input sequences
    result = sequence.pad_sequences(np.array([result]), maxlen=max_review_length)  # zero pads at the beggining of list: array([[    0,     0,     0, ..., 1, 14], [    0,     0,     0, ..., 145,    95]], dtype=int32)
    X1_enc.append(result[0])

X1_enc = np.array(X1_enc)

# split this encoded vectors
Xtrain_enc = X1_enc[:split_point]
Xvalidation_enc = X1_enc[split_point:(split_point+1301)]
Xtest_enc = X1[(split_point + 1301):-1]

#Create model, run model on different Y values
from sklearn.preprocessing import LabelBinarizer

embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary()) #compare to CNN

d = {}
labels = df.columns[:-1]
for i in labels:
    Y = df[i].tolist()
    encoder = LabelBinarizer()
    transfomed_label = encoder.fit_transform(Y)
    Ytrain = transfomed_label[:split_point]  # Should be: array([1, 0, 0, ..., 0, 1, 0])
    Yvalidation = transfomed_label[split_point:(split_point + 1301)]
    Ytest = transfomed_label[(split_point + 1301):-1]
    # if oversample:
    #     train_oversample = oversample(Xtrain, Ytrain, i) #oversample
    #     Ytrain = train_oversample.iloc[:,0]
    #     Xtrain = train_oversample.iloc[:,1]
    model.fit(Xtrain_enc, Ytrain, epochs=3, batch_size=128)  # A large batch size of 64 reviews is used to space out weight updates.
    # Final evaluation of the model
    scores = model.evaluate(Xvalidation_enc, Yvalidation, verbose=0)
    print("acc: %.2f%%" % (scores[1] * 100))
    acc = scores[1]*100
    d[i]= acc

data = pd.DataFrame(columns=d.keys())
data = data.append(pd.DataFrame(d, index=[0]), ignore_index=True)
data = data.transpose()
data.columns =['model']
print(data)

# train_oversample = oversample(Xtrain, Ytrain, i) #oversample
# Ytrain = train_oversample.iloc[:,0]
# Xtrain = train_oversample.iloc[:,1]

# create the model
# ==================================================================================================

# Run SVM model
'''
Xtrain = X1[:split_point]
labels = df.columns[:-1]
Xvalidation = X1[split_point:(split_point+1301)]
Xtest = X1[(split_point + 1301):-1]
d = {}
labels = df.columns[:-1]
for i in labels:
    Y = df[i].tolist()
    Ytrain = Y[:split_point]
    Yvalidation = Y[split_point:(split_point+1301)]
    Ytest = Y[(split_point+1301):-1]

    train_oversample = oversample(Xtrain, Ytrain, i) #oversample
    Ytrain = train_oversample.iloc[:,0]
    Xtrain = train_oversample.iloc[:,1]

    clf = LinearSVC(class_weight='balanced')
    vect = CountVectorizer(max_features=50000)
    text_clf = Pipeline([('vect', vect),
                         ('tfidf', TfidfTransformer()),
                         ('clf', clf),])
    text_clf.fit(Xtrain, Ytrain)
    Yguess = text_clf.predict(Xvalidation)
    acc = np.mean(Yguess == Yvalidation)
    f1 = f1_score(Yvalidation, Yguess, average='weighted')
    report = classification_report(Yvalidation, Yguess)
    print(report)
    d[i]=round(f1, 4) * 100
'''