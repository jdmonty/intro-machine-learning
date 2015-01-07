import pickle
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib

# shouldn't have to re-train, correct?
iris = datasets.load_iris()
X = iris.data

# load a persisted model from disk using joblib
clf = joblib.load('digit_classifier.pkl')
prediction = clf.predict(X[0])
# should be the same?
print(prediction)
