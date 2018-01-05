import numpy as np
import os
import pandas as pd
# import http_cleaner
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
import itertools
import collections
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score
import random
import sys
from sklearn.model_selection import train_test_split
from nltk.tokenize.casual import TweetTokenizer
import sys
import importlib
import random
import config
importlib.reload(config)
path = config.path_corpus
sys.path.append(path)
import split_types
import http_cleaner
from sklearn import preprocessing


plot_conf_matrix = False
print_report_for_latex = False
print_report = True
liwc = False #True = liwc, False = TFIDF

def classification_report_df(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = list(filter(None, line.split(' ')))
        print(row_data)
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    df_report = pd.DataFrame.from_dict(report_data)
    df_latex = df_report.to_latex()
    return df_latex

# Plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    if normalize:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

# Choose df
##============================================================================================
# FULL label
# df = pd.read_csv(path) #full label (e.g., INTP)
# X = df['posts'].tolist()
# Y = df['type'].tolist()

# Partial label
df = split_types.new_df  # partial label (e.g., I)
df = df.sample(frac=1)

df.to_csv(config.path+'df.csv')

X = df['posts'].tolist()

# Clean data
##============================================================
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
clear_http = http_cleaner.Mycleaner()

def preprocess1(listOfLists):
    # not removing URLs
    X1 = []
    for i in listOfLists:
        i = i.replace('|||', ' ')
        tokens = tokenizer.tokenize()
        X1.append(' '.join(tokens))
    return X1

def preprocess1(listOfLists):
	'''removing URLs'''
	X1 = []
	for i in listOfLists:
	    i = i.replace('|||', ' ')
	    tmp_tokens = []
	    tokens = tokenizer.tokenize(i)
	    for j in tokens:
	        tmp = clear_http.clean(j)
	        tmp_tokens.append(tmp)
	    X1.append(' '.join(tmp_tokens).replace('  ',' '))
	return X1


# np.save(config.path+'X1',X1)

X1 = np.load(config.path+'X1.npy').tolist()

# # Create txt files (1 per subject) with posts in 3 folders (train, validation, test), then feed to LIWC
##============================================================
# sets = [Xtrain,Xvalidation,Xtest]
# sets1 = ['train','validation','test']


# for i in range(3):
#     for text in range(len(sets[i])):
#         with open(config.path+sets1[i]+'/'+str(text).zfill(4)+".txt", "w") as text_file:
#             text_file.write(sets[i][text])


# split 0.7, 0.15, 0.15
##============================================================
split_point = int(0.7 * len(X1))

# Normalize liwc values
##============================================================

def normalize(df):
    df['WC'] = pd.to_numeric(df['WC']).astype(float)
    df.convert_objects(convert_numeric=True)
    df_norm = (df - df.mean()) / (df.max() - df.min())
    return df_norm

# Split data and run for the four labels
# ========
if liwc:
    '''liwc features'''
    liwc_train = pd.read_csv(config.path + 'liwc_train.csv')
    liwc_validation = pd.read_csv(config.path + 'liwc_validation.csv')
    liwc_train_normalized = normalize(liwc_train.iloc[:,2:])
    liwc_validation_normalized = normalize(liwc_validation.iloc[:,2:])
    Xtrain1 = np.array(liwc_train_normalized)
    Xvalidation1 = np.array(liwc_validation_normalized)
    d = {}
    labels = df.columns[:-1]
    for i in labels:
        Y = df[i].tolist()
        Ytrain = Y[:split_point]
        Yvalidation = Y[split_point:(split_point+1301)]
        Ytest = Y[(split_point+1301):-1]
        clf = LinearSVC()
        clf.fit(Xtrain1, Ytrain)
        Yguess = clf.predict(Xvalidation1)
        f1 = f1_score(Yvalidation, Yguess, average='weighted')
        report = classification_report(Yvalidation, Yguess)
        print(report)
        d[i]=round(f1, 4) * 100
else:
    '''TFIDF features'''
    Xtrain = X1[:split_point]
    Xvalidation = X1[split_point:(split_point+1301)]
    Xtest = X1[(split_point + 1301):-1]
    d = {}
    labels = df.columns[:-1]
    for i in labels:
        Y = df[i].tolist()
        Ytrain = Y[:split_point]
        Yvalidation = Y[split_point:(split_point+1301)]
        Ytest = Y[(split_point+1301):-1]
        clf = LinearSVC()
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

data = pd.DataFrame(columns=d.keys())
data = data.append(pd.DataFrame(d, index=[0]), ignore_index=True)
data = data.transpose()
data.columns =['model']
print(data)


# =========
a = ["my first post as a travel blogger in ohlala magazine!! It is one of the biggest - if not the biggest and most read magazine by women in argentina, cannot believe i am collaborating for them from sweden! great way to close the year. if you read some spanish check it out, if not you can use facebook translate! more in my travel instagram. my first post as a travel blogger in ohlala magazine! it is one of the biggest - if not the biggest and most read magazine by women in argentina, cannot believe i am collaborating for"]

a = preprocess1(a)

'''Predict personality with a given text'''

labels = df.columns[:-1]
for i in labels:
    Y = df[i].tolist()
    Ytrain = Y[:split_point]
    Ytest = Y[(split_point + 1301):-1]
    clf = LinearSVC()
    vect = CountVectorizer(max_features=50000)
    text_clf = Pipeline([('vect', vect),
                         ('tfidf', TfidfTransformer()),
                         ('clf', clf), ])
    text_clf.fit(Xtrain, Ytrain)
    Yguess = text_clf.predict(a)
    print(Yguess)


# Test a single model and method
##============================================================================================
# model = 'linearSVC'
# vectorizer = 'unigram'
#
#
# if model == 'SVM':
#     clf = SVC(kernel='linear', C = 1.0)
# elif model == 'linearSVC':
#     clf = LinearSVC()
# elif model == 'NB':
#     clf = MultinomialNB()
#
# if vectorizer == 'ngrams':
#     vect = CountVectorizer(ngram_range=(1, 3), max_features=100000)
# elif vectorizer == 'lemmatizer':
#     lemmatizer = WordNetLemmatizer()
#     def lemmatized_words(doc):
#         return (lemmatizer.lemmatize(w) for w in analyzer(doc))
#     analyzer = CountVectorizer().build_analyzer()
#     vect = CountVectorizer(analyzer=lemmatized_words)
# elif vectorizer == 'unigram':
#     vect = CountVectorizer(max_features=50000)
#
#
# text_clf = Pipeline([('vect', vect),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', clf),])
#
# text_clf.fit(Xtrain, Ytrain)
# Yguess = text_clf.predict(Xvalidation)
# acc = np.mean(Yguess == Yvalidation)
# f1 = f1_score(Yvalidation, Yguess, average='weighted')
# print(round(f1,4)*100)
#
# # Classification report
# # class_names = ['books', 'camera', 'dvd', 'health', 'music', 'software']
#
# if print_report:
#     report = classification_report(Yvalidation, Yguess)
#     print(report)
#
# if print_report_for_latex:
#     df = classification_report_df(report)
#     print(df)
#
# # Confusion Matrix
# if plot_conf_matrix == True:
#     cm = confusion_matrix(Ytest, Yguess) # Compute confusion matrix
#     plt.figure() # plot confusion matrix
#     plot_confusion_matrix(cm, np.array(class_names), normalize=True)
#     plt.tight_layout()
#     # plt.show()
#     # plt.savefig(outpath+'cm_'+model+'_'+vectorizer+'.eps',format='eps', dpi=100)
#


#For summary of results between models and methods:
#============================================================================================
# print('Computing NB and SVM in all different ways...')
#
# d3 = {}
#
# model
# vectorizer = []
#
# for i in range(6):
#     if i == 0:
#         model = 'NB'
#         vectorizer = 'unigram'
#     elif i == 1:
#         model = 'NB'
#         vectorizer = 'ngrams'
#     elif i == 2:
#         model = 'NB'
#         vectorizer = 'lemmatizer'
#     elif i == 3:
#         model = 'SVM'
#         vectorizer = 'unigram'
#     elif i == 4:
#         model = 'SVM'
#         vectorizer = 'ngrams'
#     elif i == 5:
#         model = 'SVM'
#         vectorizer = 'lemmatizer'
#     if model == 'SVM':
#         clf = LinearSVC()
#     elif model == 'NB':
#         clf = MultinomialNB()
#     if vectorizer == 'ngrams':
#         vect = CountVectorizer(ngram_range=(1, 3), max_features=100000)
#     elif vectorizer == 'lemmatizer':
#         lemmatizer = WordNetLemmatizer()
#         def lemmatized_words(doc):
#             return (lemmatizer.lemmatize(w) for w in analyzer(doc))
#         analyzer = CountVectorizer().build_analyzer()
#         vect = CountVectorizer(analyzer=lemmatized_words)
#     elif vectorizer == 'unigram':
#         vect = CountVectorizer(max_features=50000)
#
#     text_clf = Pipeline([('vect', vect),
#                          ('tfidf', TfidfTransformer()),
#                          ('clf', clf), ])
#
#
#     text_clf.fit(Xtrain, Ytrain)
#     Yguess = text_clf.predict(Xtest)
#     acc = np.mean(Yguess == Ytest)
#     f1 = f1_score(Ytest, Yguess, average='weighted')*100
#     d3[model+' '+vectorizer] = round(f1,2)
#     print(f1)
#
# data = pd.DataFrame(columns=d3.keys())
# data = data.append(pd.DataFrame(d3, index=[0]), ignore_index=True)
# data = data.transpose()
# data.columns =['model']
# print(data)

