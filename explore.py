# Data handling
import pandas as pd
import re

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt



# Import data
trainDF = pd.read_csv("input/train.csv")
testDF = pd.read_csv("input/test.csv")
combine = pd.concat([trainDF, testDF], axis=0).reset_index(drop=True)

r = re.compile('(\w*\.)')
titles = list(map(lambda x: r.search(x).group(1), combine['Name']))




# Null checking
nulls = combine.isnull().sum().sort_values(ascending=False)




# print(m.search(name).group(1))

