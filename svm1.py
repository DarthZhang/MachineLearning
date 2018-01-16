import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import f1_score
import sys
import config #python script in directory
import split_types #python script in directory
import data_cleaner
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
import cm_clas_report #python script in directory
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
import load_data

from sklearn.model_selection import train_test_split
import importlib
importlib.reload(load_data)
path = config.corpus
sys.path.append(path)

plot_conf_matrix = False
print_report_for_latex = False
print_report = True
oversample1 = False #worsens performance
features = 'tfidf' ## Select the type of features: tfidf, liwc, both
full_or_partial_label = load_data.full_or_partial_label # True = full, False = Partial
cv_gridsearch = False


def normalize(df):
    df['WC'] = pd.to_numeric(df['WC']).astype(float)
    df.convert_objects(convert_numeric=True)
    df_norm = (df - df.mean()) / (df.max() - df.min())
    return df_norm

def oversample(Xtrain, Ytrain, label):
    # Separate majority and minority classes
    df_labels = pd.Series(np.array(Ytrain))
    df_posts = pd.Series(np.array(Xtrain))
    df = pd.concat([df_labels, df_posts], axis=1)

    if label == 'type_1':
        df_majority = df[df.iloc[:, 0] == 'I']
        df_minority = df[df.iloc[:, 0] == 'E']
    elif label == 'type_2':
        df_majority = df[df.iloc[:, 0] == 'N']
        df_minority = df[df.iloc[:, 0] == 'S']
    elif label == 'type_3':
        df_majority = df[df.iloc[:, 0] == 'F']
        df_minority = df[df.iloc[:, 0] == 'T']
    elif label == 'type_4':
        df_majority = df[df.iloc[:, 0] == 'P']
        df_minority = df[df.iloc[:, 0] == 'J']
    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=df_majority.shape[0],  # to match majority class
                                     random_state=123)  # reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    return df_upsampled


# # Create txt files (1 per subject) with posts in 3 folders (train, validation, test), then feed to LIWC
##============================================================
# sets = [Xtrain,Xvalidation,Xtest]
# sets1 = ['train','validation','test']


# for i in range(3):
#     for text in range(len(sets[i])):
#         with open(config.path+sets1[i]+'/'+str(text).zfill(4)+".txt", "w") as text_file:
#             text_file.write(sets[i][text])


## Load dataset
# =====================================================================================

df = load_data.df

X1 = load_data.X1
Xtrain = load_data.Xtrain
Xtest = load_data.Xtest

if full_or_partial_label:
    Ytrain = load_data.Ytrain
    Ytest = load_data.Ytest

# run
# ========================================================================
labels = df.columns[:-1]

if features == 'tfidf':
    '''TFIDF features'''
    d = {}
    labels = df.columns[:-1]
    if full_or_partial_label: # 1 in 16 classification
        # if oversample1:
            # oversample minority classes to match majority class
            # TODO: only works with 4 binary classification
            # train_oversample = oversample(Xtrain, Ytrain, i)  # oversample
            # Ytrain = train_oversample.iloc[:, 0]
            # Xtrain = train_oversample.iloc[:, 1]
        if cv_gridsearch:
            vect = CountVectorizer()
            SVM = LinearSVC()
            pipeline = Pipeline([('vect', vect),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', SVM), ])
            parameters = {'clf__C': [0.001, 0.01, 0.1, 1, 10],
                          'clf__class_weight': [None,'balanced'],
                          'vect__max_features': [10000,50000,100000]
                          }
            clf = GridSearchCV(pipeline, parameters, cv=6, scoring='f1_weighted', refit=False, verbose=1)
            clf = clf.fit(Xtrain, Ytrain)
            scores = clf.cv_results_['mean_test_score']
            print(np.mean(scores))
            print(clf.best_score_)
            best_params = clf.best_params_
            print(best_params)
        else:
            # NO Crossvalidation, no gridsearch, use for final evaluation on test set.
            clf = LinearSVC(class_weight='balanced')
            vect = CountVectorizer(max_features=100000)
            text_clf = Pipeline([('vect', vect),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', clf), ])
            # text_clf.fit(Xtrain, Ytrain)
            scores = cross_val_score(text_clf, Xtrain, Ytrain, cv=6, scoring='f1_weighted')
            Yguess = text_clf.predict(Ytest)
            acc = np.mean(Yguess == Ytest)
            f1 = f1_score(Ytest, Yguess, average='weighted')
            # print(np.round(f1*100,2))
            report = classification_report(Ytest, Yguess)
           # print(report)
    else:
        '4 binary classification'
        for i in labels:
            Y = df[i].tolist()
            Ytrain = Y[:split_point]
            Yvalidation = Y[split_point:(split_point+1301)]
            Ytest = Y[(split_point+1301):-1]
            if oversample1:
                train_oversample = oversample(Xtrain, Ytrain, i) #oversample
                Ytrain = train_oversample.iloc[:,0]
                Xtrain = train_oversample.iloc[:,1]
            clf = LinearSVC(class_weight='balanced',C=10.0)
            # clf = SVC(C=10.0, kernel='linear',gamma=0.01)
            vect = CountVectorizer(max_features=50000)
            text_clf = Pipeline([('vect', vect),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', clf),])


            text_clf.fit(Xtrain, Ytrain)
            Yguess = text_clf.predict(Xvalidation)
            acc = np.mean(Yguess == Yvalidation)
            f1 = f1_score(Yvalidation, Yguess, average='weighted')
            print(np.round(f1 * 100, 2))
            report = classification_report(Yvalidation, Yguess)
            print(report)
            d[i]=round(f1, 4) * 100

if features == 'liwc':
    '''liwc features'''
    # TODO: crossvalidation
    liwc_train = pd.read_csv(config.path + 'liwc_train.csv')
    liwc_validation = pd.read_csv(config.path + 'liwc_validation.csv')
    liwc_train_normalized = normalize(liwc_train.iloc[:, 2:])
    liwc_validation_normalized = normalize(liwc_validation.iloc[:, 2:])
    Xtrain1 = np.array(liwc_train_normalized)
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

if features == 'both':
    '''
    concatenate LIWC and TFIDF
    '''
    # TODO: crossvalidation
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

# summarize results for all 4 models
if not full_or_partial_label:
    data = pd.DataFrame(columns=d.keys())
    data = data.append(pd.DataFrame(d, index=[0]), ignore_index=True)
    data = data.transpose()
    data.columns =['model']
    print(data)

## Run for an individual of choice
# ==========================================================================================
# a = ["my first post as a travel blogger in ohlala magazine!! It is one of the biggest - if not the biggest and most read"
#      " magazine by women in argentina, cannot believe i am collaborating for them from sweden! great way to close the "
#      "year. if you read spme spanish check it out, if not, you can use facebook translate! more in my travel instagram. "
#      "my first post as a travel blogger in ohlala magazine! it is one of the biggest - if not the biggest and most read "
#      "magazine by women in argentina, cannot believe i am collaborating for"]
#
# a = data_cleaner.preprocess2(a)
#
# '''Predict personality with a given text'''
#
# labels = df.columns[:-1]
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
#     df = cm_class_report.classification_report_df(report)
#     print(df)
#
# # Confusion Matrix
# if plot_conf_matrix == True:
#     cm = confusion_matrix(Ytest, Yguess) # Compute confusion matrix
#     plt.figure() # plot confusion matrix
#     cm_class_report.plot_confusion_matrix(cm, np.array(class_names), normalize=True)
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

