import numpy as np
import pprint as pp
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
from sklearn import neighbors
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
import sys
import importlib
import random
import config
import split_types
import data_cleaner
from sklearn import preprocessing


importlib.reload(config)
path = config.corpus
sys.path.append(path)

plot_conf_matrix = False
print_report_for_latex = False
print_report = True
liwc = False #True = liwc, False = TFIDF

# Choose df
##============================================================================================
# FULL label
# df = pd.read_csv(path) #full label (e.g., INTP)
# X = df['posts'].tolist()
# Y = df['type'].tolist()

# Partial label
df = split_types.new_df  # partial label (e.g., I)
df = df.sample(frac=1) # frac specifies a fraction of the whole dataset, e.g 0.5 -> 50%

#Save it in order to skip loading time
#df.to_csv(config.path+'df.csv')

X = df['posts'].tolist()

# Clean data
##============================================================
data_cleaner = data_cleaner.data_cleaner()

# np.save(config.path+'X1',X1)

X1 = np.load(config.path+'X1.npy').tolist()
#print(preprocess2(X))
#print(X1)
#exit(0)
split_point = int(0.7 * len(X1))

X_train = X1[:split_point]
X_validation = X1[split_point:(split_point + 1301)]
X_test = X1[(split_point + 1301):-1]
d = {}
labels = df.columns[:-1]

for i in labels:
    Y = df[i].tolist()
    Y_train = Y[:split_point]
    Y_validation = Y[split_point:(split_point + 1301)]
    Y_test = Y[(split_point + 1301):-1]
    clf = LinearSVC()
    #vect = CountVectorizer(max_features=50000)

    knn = neighbors.KNeighborsClassifier()
    dt = tree.DecisionTreeClassifier()
    svm = SVC()

    pipelines = [Pipeline([('knn', knn)]), Pipeline([('dt', dt)]), Pipeline([('svm', svm)])]
    parameters = {
        'knn__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
        'knn__weights': ['uniform', 'distance'],
        'knn__algorithm': [ 'ball_tree', 'kd_tree', 'brute']
    }

    parameters2 = {
        'dt__criterion': ['gini', 'entropy'],
        'dt__max_leaf_nodes':[2,3,4,5],
        'dt__max_depth': [2,3,4,5],
        'dt__splitter': ['best']  # Use random it brings to a non certain result
    }

    parameters3 = {
        'svm__C': [.001, .01, .1, 1.0, 10.],
        'svm__kernel': ['rbf', 'linear', 'sigmoid'],
        'svm__gamma': [.001, .01, .1, 1.0]
    }

    final_parameters = [parameters, parameters2, parameters3]

    for (pip, pr) in zip(pipelines, final_parameters):
        grid_search = GridSearchCV(pip,
                                   pr,
                                   scoring=metrics.make_scorer(metrics.matthews_corrcoef),
                                   cv=10,
                                   n_jobs=-1,
                                   verbose= 0)
        grid_search.fit(X_train, Y_train)

        #  Print results for each combination of parameters.
        number_of_candidates = len(grid_search.cv_results_['params'])
        print("Results:")
        for i in range(number_of_candidates):
            print(i, 'pa-rams - %s; mean - %0.3f; std - %0.3f' %
                  (grid_search.cv_results_['params'][i],
                   grid_search.cv_results_['mean_test_score'][i],
                   grid_search.cv_results_['std_test_score'][i]))

        print("Best Estimator:")
        pp.pprint(grid_search.best_estimator_)

        print("Best Parameters:")
        pp.pprint(grid_search.best_params_)

        print("Used Scorer Function:")
        pp.pprint(grid_search.scorer_)

        print("Number of Folds:")
        pp.pprint(grid_search.n_splits_)

        Y_predicted = grid_search.predict(X_test)

        output_classification_report = metrics.classification_report(
            Y_test,
            Y_predicted)

        print("----------------------------------------------------")
        print(output_classification_report)
        print("----------------------------------------------------")

        # Compute the confusion matrix
        confusion_matrix = metrics.confusion_matrix(Y_test, Y_predicted)

        print("Confusion Matrix: True-Classes X Predicted-Classes")
        print(confusion_matrix)

        print("Matthews corrcoefficent")
        print(metrics.matthews_corrcoef(Y_test, Y_predicted))

        print("Normalized Accuracy")
        print(metrics.accuracy_score(Y_test, Y_predicted))
