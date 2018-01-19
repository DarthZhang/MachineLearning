from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection

from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
import load_data_from_zero

#Load data
df = load_data_from_zero.df
X = load_data_from_zero.X1


d = {}
labels = df.columns[:-1]

for i in labels:
    Y = df[i]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, train_size=0.8, random_state=42)

    vectorizer = TfidfVectorizer(strip_accents=None, preprocessor=None, stop_words='english')

    estimators = []
    model1 = SVC(C = 10.0, kernel= 'rbf', gamma = 0.1)
    estimators.append(('svm10rbf0.1', model1))
    # model2 = tree.DecisionTreeClassifier(criterion='gini', max_depth=2, max_leaf_nodes=4, splitter='best')
    # estimators.append(('dt_gini24', model2))
    model3 = SVC(C = 10.0, kernel= 'linear', gamma = 0.01)
    estimators.append(('svm10linear0.01', model3))
    model4 = SVC(C = 1.0, kernel= 'linear', gamma = 0.01)
    estimators.append(('svm1linear0.01', model4))
    model5 = AdaBoostClassifier(algorithm='SAMME.R')
    estimators.append((('adb'),model5))


    ensemble = VotingClassifier(estimators)

    pipeline= Pipeline([('vect', vectorizer), ('ens', ensemble)])

    parameters = {
        'ens__voting': ['hard'],
    }

    grid_search = GridSearchCV(pipeline,
                               parameters,
                               scoring=metrics.make_scorer(metrics.matthews_corrcoef),
                               cv=10,
                               n_jobs=-1,
                               verbose= 10)


    grid_search.fit(X_train, Y_train)

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