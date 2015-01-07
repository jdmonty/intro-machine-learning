from sklearn import datasets
from sklearn import svm

# key terms: training set vs testing set
iris = datasets.load_iris()
digits = datasets.load_digits()

# in this case the digits image/classifier or predictor
# we have
# - Samples of 10 possible classes (digitis 0 - 9)
# - Apply or "Fit Estimator" to predict classes or classify

# Existing in lib is a Support Vector Classification impl
# sklearn.svm.SVC

# Here we set the parameters manually
# TIP/NOTE: You can find params automatically (automagically)
# using tools suchas "grid search" and "cross validation"
clf = svm.SVC(gamma=0.001, C=100)

# pass training set to "fit" method
# arbitrary example... all but last digit
estimator = clf.fit(digits.data[:-1], digits.target[:-1])
print(estimator)

# now the prediction/magic
prediction = clf.predict(digits.data[-1])
print(prediction)
