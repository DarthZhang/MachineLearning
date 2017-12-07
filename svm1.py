import numpy as np
import os
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import os
import pandas as pd
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
import os
import sys
from sklearn.model_selection import train_test_split

# Alternative paths
path = '/Users/danielmlow/Dropbox/lct/ml/project/'
outpath = '/Users/danielmlow/Dropbox/lct/lfd/a3/'
corpus = 'mbti_1.csv'


df = pd.read_csv(path+corpus)







# Parameters
model = 'SVM' # SVM or NB or linearSVM
vectorizer = 'unigram' # unigram, ngrams or lemmatizer
plot_conf_matrix = False

# Open corpus, obtain data and and labels: if parameter "use_sentiment=True", then append labels= pos or neg, else labels=1 of 6 topics






def read_corpus(corpus_file):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            documents.append(' '.join(tokens[3:]))
            labels.append(tokens[0])
    return documents, labels

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



# create 2 mock train and test daa sets.
##============================================================================================
# with open(path+corpus, encoding='utf-8') as f:
#     content = f.readlines()
#
# random.shuffle(content)
#
# split_point = int(0.75 * len(X))
# train_data = content[:split_point]
# test_data = content[split_point:]
#
# with open(outpath+'train_set_mock.txt','w') as f:
#     for line in train_data:
#         f.write(str(line))
# with open(outpath+'test_set_mock.txt','w') as f:
#     for line in test_data:
#         f.write(str(line))


# Load datasets from command line:
##============================================================================================
# Xtrain, Ytrain= read_corpus(train_path)
# Xtest, Ytest = read_corpus(test_path)

# Or Load corpus, perform 75-25 train-test set split
##============================================================================================
# X, Y = read_corpus(path + corpus)

X = df['posts'].tolist()

Y = df['type'].tolist()

split_point = int(0.7 * len(X))
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xvalidation = X[split_point:(split_point+1301)]
Yvalidation = Y[split_point:(split_point+1301)]
Xtest = X[(split_point+1301):-1]
Ytest = Y[(split_point+1301):-1]



# Test a single model and method
##============================================================================================
model = 'SVM'
vectorizer = 'ngrams'



if model == 'SVM':
    clf = SVC(kernel='linear', C = 1.0)
elif model == 'linearSVC':
    clf = LinearSVC()
elif model == 'NB':
    clf = MultinomialNB()

if vectorizer == 'ngrams':
    vect = CountVectorizer(ngram_range=(1, 3), max_features=100000)
elif vectorizer == 'lemmatizer':
    lemmatizer = WordNetLemmatizer()
    def lemmatized_words(doc):
        return (lemmatizer.lemmatize(w) for w in analyzer(doc))
    analyzer = CountVectorizer().build_analyzer()
    vect = CountVectorizer(analyzer=lemmatized_words)
elif vectorizer == 'unigram':
    vect = CountVectorizer(max_features=50000)


text_clf = Pipeline([('vect', vect),
                     ('tfidf', TfidfTransformer()),
                     ('clf', clf),])

text_clf.fit(Xtrain, Ytrain)
Yguess = text_clf.predict(Xvalidation)
acc = np.mean(Yguess == Yvalidation)
f1 = f1_score(Yvalidation, Yguess, average='weighted')
print(round(f1,4)*100)

# Classification report
class_names = ['books', 'camera', 'dvd', 'health', 'music', 'software']
report = classification_report(Ytest, Yguess,target_names=class_names)
print(report)
df = classification_report_df(report)
# print(df)

# Confusion Matrix
if plot_conf_matrix == True:
    cm = confusion_matrix(Ytest, Yguess) # Compute confusion matrix
    plt.figure() # plot confusion matrix
    plot_confusion_matrix(cm, np.array(class_names), normalize=True)
    plt.tight_layout()
    # plt.show()
    # plt.savefig(outpath+'cm_'+model+'_'+vectorizer+'.eps',format='eps', dpi=100)



For summary of results between models and methods:
#============================================================================================
print('Computing NB and SVM in all different ways...')

d3 = {}

model
vectorizer = []

for i in range(6):
    if i == 0:
        model = 'NB'
        vectorizer = 'unigram'
    elif i == 1:
        model = 'NB'
        vectorizer = 'ngrams'
    elif i == 2:
        model = 'NB'
        vectorizer = 'lemmatizer'
    elif i == 3:
        model = 'SVM'
        vectorizer = 'unigram'
    elif i == 4:
        model = 'SVM'
        vectorizer = 'ngrams'
    elif i == 5:
        model = 'SVM'
        vectorizer = 'lemmatizer'
    if model == 'SVM':
        clf = LinearSVC()
    elif model == 'NB':
        clf = MultinomialNB()
    if vectorizer == 'ngrams':
        vect = CountVectorizer(ngram_range=(1, 3), max_features=100000)
    elif vectorizer == 'lemmatizer':
        lemmatizer = WordNetLemmatizer()
        def lemmatized_words(doc):
            return (lemmatizer.lemmatize(w) for w in analyzer(doc))
        analyzer = CountVectorizer().build_analyzer()
        vect = CountVectorizer(analyzer=lemmatized_words)
    elif vectorizer == 'unigram':
        vect = CountVectorizer(max_features=50000)

    text_clf = Pipeline([('vect', vect),
                         ('tfidf', TfidfTransformer()),
                         ('clf', clf), ])


    text_clf.fit(Xtrain, Ytrain)
    Yguess = text_clf.predict(Xtest)
    acc = np.mean(Yguess == Ytest)
    f1 = f1_score(Ytest, Yguess, average='weighted')*100
    d3[model+' '+vectorizer] = round(f1,2)
    print(f1)

data = pd.DataFrame(columns=d3.keys())
data = data.append(pd.DataFrame(d3, index=[0]), ignore_index=True)
data = data.transpose()
data.columns =['model']
print(data)