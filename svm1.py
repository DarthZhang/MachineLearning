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
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score
import random
import sys
from sklearn.model_selection import train_test_split
import sys
import importlib
import config
import split_types
import data_cleaner
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer


# importlib.reload(config)
path = config.corpus
sys.path.append(path)

plot_conf_matrix = False
print_report_for_latex = False
print_report = True
oversample1 = False

# Select the type of features
features = 'tfidf' #tfidf, liwc, both

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
df = df.sample(frac=1, random_state=123) # should randomize whole dataframe)

# Clean data
##============================================================
data_cleaner = data_cleaner.data_cleaner()

## First time you run:
# X = df['posts'].tolist()
# print('preprocessing corpus...')
# X1 = data_cleaner.preprocess1(X)
# print('Done preprocessing')
# np.save(config.path+'X2',X1)

# After:
X1 = np.load(config.path+'X2.npy').tolist()
# X1 = np.load(config.path+'X1.npy')

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


labels = df.columns[:-1]
'''liwc features'''
liwc_train = pd.read_csv(config.path + 'liwc_train.csv')
liwc_validation = pd.read_csv(config.path + 'liwc_validation.csv')
liwc_train_normalized = normalize(liwc_train.iloc[:, 2:])
liwc_validation_normalized = normalize(liwc_validation.iloc[:, 2:])
if features == 'liwc':
    Xtrain1 = np.array(liwc_train_normalized)
    # print Xtrain1.shape
    Xvalidation1 = np.array(liwc_validation_normalized)
    d = {}
    for i in labels:
        Y = df[i].tolist()
        Ytrain = Y[:split_point]
        Yvalidation = Y[split_point:(split_point+1301)]
        Ytest = Y[(split_point+1301):-1]
        clf = LinearSVC(class_weight='balanced')
        clf.fit(Xtrain1, Ytrain)
        Yguess = clf.predict(Xvalidation1)
        f1 = f1_score(Yvalidation, Yguess, average='weighted')
        report = classification_report(Yvalidation, Yguess)
        print(report)
        d[i]=round(f1, 4) * 100
elif features == 'tfidf':
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
        if oversample1:
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
elif features == 'both':
    Xtrain_liwc = np.array(liwc_train_normalized)
    Xvalidation_liwc = np.array(liwc_validation_normalized)

    Xtrain_texts = X1[:split_point]
    Xvalidation_texts = X1[split_point:(split_point+1301)]
    vectorizer = TfidfVectorizer(min_df=1, stop_words="english", max_features=50000)
    Xtrain_tfidf = vectorizer.fit_transform(Xtrain_texts).toarray()
    Xvalidation_tfidf = vectorizer.fit_transform(Xvalidation_texts).toarray()
    print(Xtrain_tfidf.shape)
    print(Xvalidation_tfidf.shape)

    # liwc_train_normalized = normalize(liwc_train.iloc[:, 2:])
    for i in labels:
        print('classification report')




data = pd.DataFrame(columns=d.keys())
data = data.append(pd.DataFrame(d, index=[0]), ignore_index=True)
data = data.transpose()
data.columns =['model']
print(data)


# =========
a = ["my first post as a travel blogger in ohlala magazine!! It is one of the biggest - if not the biggest and most read"
     " magazine by women in argentina, cannot believe i am collaborating for them from sweden! great way to close the "
     "year. if you read spme spanish check it out, if not, you can use facebook translate! more in my travel instagram. "
     "my first post as a travel blogger in ohlala magazine! it is one of the biggest - if not the biggest and most read "
     "magazine by women in argentina, cannot believe i am collaborating for"]

a = data_cleaner.preprocess1(a)

'''Predict personality with a given text'''

labels = df.columns[:-1]
# for i in labels:
#     Y = df[i].tolist()
#     Ytrain = Y[:split_point]
#     Ytest = Y[(split_point + 1301):-1]
#     clf = LinearSVC()
#     vect = CountVectorizer(max_features=50000)
#     text_clf = Pipeline([('vect', vect),
#                          ('tfidf', TfidfTransformer()),
#                          ('clf', clf), ])
#     text_clf.fit(Xtrain, Ytrain)
#     Yguess = text_clf.predict(a)
#     print("Example Gueess:")
#     print(Yguess)


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

