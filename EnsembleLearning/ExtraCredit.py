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


baggedTrees = Bagging.constructBaggedTree(Train, 'col16', 1, 'IG', 100)
x, y = Bagging.testTrees(baggedTrees, Test, 'col16')
print("Iteration for Individual Bagged Trees on Test Extra Credit")
for i in range(len(x)):
    print(str(x[i]))
print("Error ")
for i in range(len(y)):
    print(str(y[i]))


x, y = Bagging.testGroupDecision(baggedTrees, Test, 'col16')
print("Iteration for Individual Bagged Trees on Test Extra Credit")
for i in range(len(x)):
    print(str(x[i]))
print("Error ")
for i in range(len(y)):
    print(str(y[i]))





