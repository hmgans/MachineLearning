import math
from numpy.lib.function_base import append
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

Amount = 100
numberInBag = 5

totalBags = []
for i in range(Amount):
    newSample = Bagging.GetSamplesWithoutReplacement(1000, df)
    baggedTrees = Bagging.constructBaggedTree(newSample, 'col16', 0, 'IG', numberInBag)
    totalBags.append(baggedTrees)


# Now calute the bias term for each tree
biasTerms = []
varianceTerms = []

for i in range(numberInBag):
    bias = []
    treeVariance = []
    for j in range(Amount):
        error = Bagging.testTree(totalBags[j][i], dfTest, 'col16')
        error /= len(dfTest) # Mean
        variance = Bagging.findVariance(dfTest, error)
        treeVariance.append(variance)

        treeBias = math.pow(error-1,2) # Bias
        bias.append(treeBias)
    totalBias = 0
    totalVariance = 0
    for j in range(len(bias)):
        totalBias += bias[j]
        totalVariance += treeVariance[j]
    

    biasTerms.append(totalBias/len(bias))
    varianceTerms.append(totalVariance/len(treeVariance))

for i in range(len(biasTerms)):
    print("Tree at " + str(i)+ " bias: " + str(biasTerms[i]))
    print("Tree at " + str(i)+ " variance: " + str(varianceTerms[i]))
    print("Squared: " + str(biasTerms[i]+varianceTerms[i]))


groupBias = []
groupVariance = []

for j in range(Amount):
    error =  Bagging.testGroupMean(totalBags[j], dfTest, 'col16')
    variance = Bagging.findVariance(dfTest, error)
    treeBias =  treeBias = math.pow(error-1,2) # Bias
    groupBias.append(treeBias)
    groupVariance.append(variance)

for i in range(len(groupBias)):
    print("Tree Ensemble at " + str(i)+ " bias:" + str(groupBias[i]))
    print("Tree Ensemble at " + str(i)+ " variance:" + str(groupVariance[i]))
    print("Squared: " + str(groupBias[i]+groupVariance[i]))


    

        




        



