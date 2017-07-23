# Data handling
import pandas as pd

# Visualization
#import seaborn as sns
# import matplotlib.pyplot as plt

# Machine Learning
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

# Other
from tabulate import tabulate
import operator

trainDF = pd.read_csv("input/train.csv")
testDF = pd.read_csv("input/test.csv")
combine = [trainDF, testDF]

droppedFields = ['PassengerId', 'Name', 'Ticket', 'Cabin']
for field in droppedFields:
    trainDF.drop(field, 1, inplace=True)
    testDF.drop(field, 1, inplace=True)

print(trainDF.columns.values)


def fill_missing_ages(dataframe):
    dataframe['Age'] = dataframe.groupby(['Sex', 'Pclass']).transform(lambda x: x.fillna(x.mean()))['Age']

fill_missing_ages(trainDF)
fill_missing_ages(testDF)


def create_dummies(dataframe, field):
    dummies = pd.get_dummies(dataframe[field], drop_first=True)
    dataframe.drop(field, axis=1, inplace=True)
    return pd.concat([dataframe, dummies], axis=1)

trainDF = create_dummies(trainDF, 'Sex')
testDF = create_dummies(testDF, 'Sex')
trainDF = create_dummies(trainDF, 'Pclass')
testDF = create_dummies(testDF, 'Pclass')
trainDF = create_dummies(trainDF, 'Embarked')
testDF = create_dummies(testDF, 'Embarked')

X_train = trainDF.drop('Survived', axis=1)
X_train = preprocessing.scale(X_train)
Y_train = trainDF['Survived']


scores = {}

# Logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
scores['Logistic regression'] = logreg.score(X_train, Y_train)

# Decision tree
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
scores['Decision tree'] = model.score(X_train, Y_train)

# SVM
model = SVC(cache_size=1000)
model.fit(X_train, Y_train)
scores['SVM'] = model.score(X_train, Y_train)

# KNN
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
scores['KNN'] = model.score(X_train, Y_train)

# Random Forest
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, Y_train)
scores['Random Forest'] = model.score(X_train, Y_train)

# Gradient Boosting
model = GradientBoostingClassifier()
model.fit(X_train, Y_train)
scores['Gradient Boosting'] = model.score(X_train, Y_train)

# Perceptron
model = Perceptron()
model.fit(X_train, Y_train)
scores['Perceptron'] = model.score(X_train, Y_train)

# MLP
model = MLPClassifier(hidden_layer_sizes=30)
model.fit(X_train, Y_train)
scores['MLP'] = model.score(X_train, Y_train)


headers = ['Name', 'Score']
scores = sorted(scores.items(), key=operator.itemgetter(1))
scores.reverse()
print(tabulate(scores, headers=headers))
