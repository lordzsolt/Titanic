# Data handling
import pandas as pd

# Machine Learning
from sklearn.ensemble import RandomForestClassifier

# Other
from tabulate import tabulate
import operator

originalTrainDF = pd.read_csv("input/train.csv")
trainDF = originalTrainDF.copy()

originalTestDF = pd.read_csv("input/test.csv")
testDF = originalTestDF.copy()

droppedFields = ['PassengerId', 'Name', 'Ticket', 'Cabin']
for field in droppedFields:
    trainDF.drop(field, 1, inplace=True)
    testDF.drop(field, 1, inplace=True)

print(trainDF.columns.values)


def fill_missing(dataframe, field):
    dataframe[field] = dataframe.groupby(['Sex', 'Pclass']).transform(lambda x: x.fillna(x.mean()))[field]

fill_missing(trainDF, 'Age')
fill_missing(testDF, 'Age')
fill_missing(trainDF, 'Fare')
fill_missing(testDF, 'Fare')


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
Y_train = trainDF['Survived']
X_test = testDF

scores = {}

# Random Forest
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, Y_train)
scores['Random Forest'] = model.score(X_train, Y_train)

Y_pred = model.predict(X_test)

headers = ['Name', 'Score']
scores = sorted(scores.items(), key=operator.itemgetter(1))
scores.reverse()
print(tabulate(scores, headers=headers))

submission = pd.DataFrame({
    'PassengerId': originalTestDF['PassengerId'],
    'Survived': Y_pred
})

submission.to_csv("output/submission2.csv", index=False)
