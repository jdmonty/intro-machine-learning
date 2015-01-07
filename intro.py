from sklearn import svm
from sklearn import datasets
import pickle

# build and store it
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
svc = clf.fit(X, y)
print(svc)

# dump it to string!!!
# or "pickle" it to a string?
model = pickle.dumps(clf)

#load it from the "pickle"...or string
pickled_clf = pickle.loads(model)
crystall_ball = pickled_clf.predict(X[0])
print(crystall_ball)
