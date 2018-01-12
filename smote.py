from collections import Counter
from imblearn.over_sampling import SMOTE
import load_data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import f1_score
from  sklearn import metrics

#Load data
df = load_data.df
X = load_data.X1

d = {}
labels = df.columns[:-1]
for i in labels:
    Y = df[i]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    ##============================================================

    print('Original dataset shape {}'.format(Counter(Y_train)))
    sm = SMOTE(random_state=42)
    vectorizer = CountVectorizer(min_df=1, max_features=50000)
    X_train = vectorizer.fit_transform(X_train)
    Xtrain, Ytrain = sm.fit_sample(X_train, Y_train)
    print('Resampled dataset shape {}'.format(Counter(Ytrain)))

    ##============================================================

    clf = LinearSVC(class_weight='balanced')

    clf.fit(Xtrain, Ytrain)
    Xvalidation = vectorizer.fit_transform(X_test)
    Y_predicted = clf.predict(Xvalidation)

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
    f1 = f1_score(Y_test, Y_predicted, average='weighted')
    d[i]=round(f1, 4) * 100
print(d)

