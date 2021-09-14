# This file uses ML decision tree techiniques
# Author: Hank Gansert
# Date: 09/11/21
import math
import pandas as pd
from pandas.core.base import NoNewAttributesMixin

df = pd.read_csv("TesterDT.csv", header=None)

length = df.loc[0].size
labels = []

for i in range(length):
    labels.append("col" + str(i))

print(labels)
df.columns = labels
print(df.loc[df['col1'] > 0])


def entropy(PropPos):
    PropNeg = 1 - PropPos
    return -PropPos * math.log2(PropPos) - PropNeg * math.log2(PropNeg)

def giniIndex(PropPos):
    PropNeg = 1 - PropPos
    return 1 - (math.pow(PropPos,2) + math.pow(PropNeg,2)) 

def decisionTree(filename, columnName, positiveIndicator):
    return 0

print( giniIndex(.25) )