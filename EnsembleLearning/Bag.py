

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



baggedTrees = Bagging.constructBaggedTree(df, 'col16', 3, 'IG', 500)

x, y = Bagging.testTrees(baggedTrees, df, 'col16')
plt.plot(x, y)
# naming the x axis
plt.xlabel('Iteration')
# naming the y axis
plt.ylabel('Error')
 
# giving a title to my graph
plt.title('Bagged Tree Individual on Training')
# function to show the plot
plt.show()

x, y = Bagging.testGroupDecision(baggedTrees, df, 'col16')
plt.plot(x, y)
# naming the x axis
plt.xlabel('Iteration')
# naming the y axis
plt.ylabel('Error')
 
# giving a title to my graph
plt.title('Bagged Tree Ensemble on Training')
# function to show the plot
plt.show()

#Results against Test
x, y = Bagging.testTrees(baggedTrees, dfTest, 'col16')
plt.plot(x, y)
# naming the x axis
plt.xlabel('Iteration')
# naming the y axis
plt.ylabel('Error')
 
# giving a title to my graph
plt.title('Bagged Tree Individual on Test')
# function to show the plot
plt.show()

x, y = Bagging.testGroupDecision(baggedTrees, dfTest, 'col16')
plt.plot(x, y)
# naming the x axis
plt.xlabel('Iteration')
# naming the y axis
plt.ylabel('Error')
# giving a title to my graph
plt.title('Bagged Tree Ensemble on Test')
# function to show the plot
plt.show()