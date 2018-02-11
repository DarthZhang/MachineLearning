import pprint as pp
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import neighbors
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import load_data_from_zero
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
#Load data
X = load_data_from_zero.X1
Y = load_data_from_zero.Y


label = pd.DataFrame(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,train_size=0.8, random_state=42)


vectorizer = TfidfVectorizer(strip_accents=None, preprocessor=None)

rf = RandomForestClassifier()
svm = SVC()
knn = neighbors.KNeighborsClassifier()
adb = AdaBoostClassifier()


pipelines =  [Pipeline([('vect', vectorizer),('knn', knn),]),
             Pipeline([('vect', vectorizer), ('rf', rf)]),
            Pipeline([('vect', vectorizer), ('svm', svm)]),
            Pipeline([('vect',vectorizer), ('adb', adb)])]
parameters = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 13],
    'knn__weights': ['uniform', 'distance'],
    'knn__algorithm': [  'brute'],
    'vect__stop_words': ['english']
}

parameters2 = {
 'rf__criterion': ['gini','entropy'],
 'rf__max_depth': [None ,4,5,6,7],
 'rf__splitter': ['best'],  # Use random it brings to a non certain result
 'rf__warm_start': [True, False],
 'vect__stop_words': [None, 'english']
}

parameters3 = {
    'vect__stop_words': ['english'],
    'svm__C': [  1.0, 10.],
    'svm__kernel': ['rbf', 'linear'],
    'svm__gamma': [ .01, .1]
}


parameters4 = {
 'vect__stop_words': [None, 'english'],
 'adb__algorithm': ['SAMME', 'SAMME.R']
}

final_parameters = [parameters, parameters2,parameters3,parameters4]

for (pip, pr) in zip(pipelines, final_parameters):
    grid_search = GridSearchCV(pip,
                               pr,
                               scoring=metrics.make_scorer(metrics.matthews_corrcoef),
                               cv=10,
                               n_jobs=-1,
                               verbose= 10)
    print("here")
    grid_search.fit(X_train, Y_train)
    print("here2")
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

    print("F1 corrcoefficent")
    print(metrics.f1_score(Y_test, Y_predicted, average="weighted"))

    print("Normalized Accuracy")
    print(metrics.accuracy_score(Y_test, Y_predicted))
