
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
randomForest = Bagging.constructRandomForestTree(df, 'col16', 2, 'IG', 500)

#Results against Test
x, y = Bagging.testTrees(randomForest, dfTest, 'col16')
plt.plot(x, y)
# naming the x axis
plt.xlabel('Iteration')
# naming the y axis
plt.ylabel('Error')
 
# giving a title to my graph
plt.title('Bagged Trees Individual on Test')
# function to show the plot
plt.show()


x, y = Bagging.testGroupDecision(randomForest, dfTest, 'col16')
print("Iteration for Group Random Forest Trees on Test")
for i in range(len(x)):
    print(str(x[i]))
print("Error ")
for i in range(len(y)):
    print(str(y[i]))