
import math
import pandas as pd
import numpy as np
from random import randrange
import random
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame

import Bagging

df = pd.read_csv("bank/train.csv", header=None)
dfTest = pd.read_csv("bank/test.csv", header=None)


df = Bagging.cleanNumbericalValues(df)
dfTest = Bagging.cleanNumbericalValues(dfTest)


#2.d
randomForest = Bagging.constructRandomForestTree(df, 'col16', 3, 'IG', 1)

#Results against Test
x, y = Bagging.testTrees(randomForest, dfTest, 'col16')
print("Iteration for Individual Random Forest Trees on Test")
for i in range(len(x)):
    print(str(x[i]))
print("Error ")
for i in range(len(y)):
    print(str(y[i]))

x, y = Bagging.testGroupDecision(randomForest, dfTest, 'col16')
print("Iteration for Group Random Forest Trees on Test")
for i in range(len(x)):
    print(str(x[i]))
print("Error ")
for i in range(len(y)):
    print(str(y[i]))