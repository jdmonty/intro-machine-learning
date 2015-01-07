from sklearn import datasets

# key terms: training set vs testing set
iris = datasets.load_iris()
digits = datasets.load_digits()

print(digits.data)

print(digits.target)
