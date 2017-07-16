# Data handling
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt


# Import data
trainDF = pd.read_csv("input/train.csv")
testDF = pd.read_csv("input/test.csv")
combine = pd.concat([trainDF, testDF], axis=0).reset_index(drop=True)


# Null checking
nulls = combine.isnull().sum().sort_values(ascending=False)
nulls = nulls[nulls > 0]
print('Null checking: \n', nulls)


# Correlations
g = sns.heatmap(trainDF[['Survived', 'SibSp', 'Parch', 'Age', 'Fare']].corr(), annot=True, cmap='coolwarm')


