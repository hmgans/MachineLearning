
import math
import pandas as pd
import numpy as np
from random import randrange
import random
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame

import Bagging

df = pd.read_csv("/Users/hankgansert/Desktop/ML/MachineLearning/EnsembleLearning/bank/train.csv", header=None)
dfTest = pd.read_csv("/Users/hankgansert/Desktop/ML/MachineLearning/EnsembleLearning/bank/test.csv", header=None)






#2.d
randomForest = Bagging.constructRandomForestTree(df, 'col16', 4, 'IG', 500)

x, y = Bagging.testTrees(randomForest, df, 'col16')
plt.plot(x, y)
# naming the x axis
plt.xlabel('Iteration')
# naming the y axis
plt.ylabel('Error')
 
# giving a title to my graph
plt.title('Random Forest Individual on Training')
# function to show the plot
plt.show()

x, y = Bagging.testGroupDecision(randomForest, df, 'col16')
plt.plot(x, y)
# naming the x axis
plt.xlabel('Iteration')
# naming the y axis
plt.ylabel('Error')
 
# giving a title to my graph
plt.title('Random Forest Ensemble on Training')
# function to show the plot
plt.show()

#Results against Test
x, y = Bagging.testTrees(randomForest, dfTest, 'col16')
plt.plot(x, y)
# naming the x axis
plt.xlabel('Iteration')
# naming the y axis
plt.ylabel('Error')
 
# giving a title to my graph
plt.title('Random Forest Individual on Test')
# function to show the plot
plt.show()

x, y = Bagging.testGroupDecision(randomForest, dfTest, 'col16')
plt.plot(x, y)
# naming the x axis
plt.xlabel('Iteration')
# naming the y axis
plt.ylabel('Error')
# giving a title to my graph
plt.title('Random Forest Ensemble on Test')
# function to show the plot
plt.show()