# This file uses ML decision tree techiniques
# Author: Hank Gansert
# Date: 09/11/21
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



#2.a 
#Get all trees and their alpha values
trees, alphas = Bagging.constructAdaTree(df, 'col16', 0, 'IG', 500)
#Results against training 
x, y = Bagging.testTrees(trees, df, 'col16')
plt.plot(x, y)
# naming the x axis
plt.xlabel('Iteration')
# naming the y axis
plt.ylabel('Error')
 
# giving a title to my graph
plt.title('AdaBoost Individual on Training')
# function to show the plot
plt.show()

x, y = Bagging.testEnsemble(trees, df, 'col16', alphas)
plt.plot(x, y)
# naming the x axis
plt.xlabel('Iteration')
# naming the y axis
plt.ylabel('Error')
 
# giving a title to my graph
plt.title('AdaBoost Ensemble on Training')
# function to show the plot
plt.show()

#Results against Test
x, y = Bagging.testTrees(trees, dfTest, 'col16')
plt.plot(x, y)
# naming the x axis
plt.xlabel('Iteration')
# naming the y axis
plt.ylabel('Error')
 
# giving a title to my graph
plt.title('AdaBoost Individual on Test')
# function to show the plot
plt.show()

x, y = Bagging.testEnsemble(trees, dfTest, 'col16', alphas)
plt.plot(x, y)
# naming the x axis
plt.xlabel('Iteration')
# naming the y axis
plt.ylabel('Error')
# giving a title to my graph
plt.title('AdaBoost Ensemble on Test')
# function to show the plot
plt.show()