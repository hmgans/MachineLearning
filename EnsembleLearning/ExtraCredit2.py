import math
import pandas as pd
import numpy as np
from random import randrange
import random
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import Bagging

df = pd.read_csv("default_of_credit_card_clients.csv", header=None)


df = Bagging.cleanNumbericalValues(df)

Train, Test = Bagging.GetSamplesWithoutReplacementTrainTest(24000, df)


randomForest = Bagging.constructRandomForestTree(df, 'col16', 3, 'IG', 100)
x, y = Bagging.testTrees(randomForest, Test, 'col16')
print("Iteration for Individual Random Forest Trees on Test Extra Credit")
for i in range(len(x)):
    print(str(x[i]))
print("Error ")
for i in range(len(y)):
    print(str(y[i]))


x, y = Bagging.testGroupDecision(randomForest, Test, 'col16')
print("Iteration for Individual Random Forest Trees on Test Extra Credit")
for i in range(len(x)):
    print(str(x[i]))
print("Error ")
for i in range(len(y)):
    print(str(y[i]))

