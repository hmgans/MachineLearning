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



baggedTrees = Bagging.constructBaggedTree(df, 'col16', 3, 'IG', 1)

x, y = Bagging.testTrees(baggedTrees, df, 'col16')
print("Iteration for Individual Bagged Trees on Train")
for i in range(len(x)):
    print(str(x[i]))
print("Error ")
for i in range(len(y)):
    print(str(y[i]))


x, y = Bagging.testGroupDecision(baggedTrees, df, 'col16')
print("Iteration for All Bagged Trees on Train")
for i in range(len(x)):
    print(str(x[i]))
print("Error ")
for i in range(len(y)):
    print(str(y[i]))

