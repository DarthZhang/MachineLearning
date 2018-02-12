from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from  sklearn import metrics
import load_data_from_zero
import pandas as pd


X = load_data_from_zero.X1
Y = load_data_from_zero.Y

label = pd.DataFrame(Y)


d = {}
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8, random_state=42)


vectorizer = CountVectorizer(stop_words='english', max_features=60000)
X_test = vectorizer.fit_transform(X_test)


##============================================================

print('Original dataset shape {}'.format(Counter(Y_train)))
sm = SMOTE(random_state=42)
X_train = vectorizer.fit_transform(X_train)
X_train, Y_train = sm.fit_sample(X_train, Y_train)
print('Resampled dataset shape {}'.format(Counter(Y_train)))

##============================================================

clf = SVC(C=10, gamma=0.1,kernel='rbf')

clf.fit(X_train, Y_train)
Y_predicted = clf.predict(X_test)

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

