import math
import pandas as pd
import numpy as np
from random import randrange
import random
import sklearn as sk
from sklearn import tree
import matplotlib.pyplot as plt
import csv


# This function will map string values to a unique number
def cleanStringValues(data):
    for key in data.keys():
        values = data[key].unique()
        mapper = {}
        i = 0
        for value in values:
            mapper[value] = i
            i += 1
        data = data.applymap(lambda s: mapper.get(s) if s in mapper else s)
    return data

def cleanStringValuesTest(data):

    data = data.drop(['ID'],axis=1)
    for key in data.keys():
        values = data[key].unique()
        mapper = {}
        i = 0
        for value in values:
            mapper[value] = i
            i += 1
        data = data.applymap(lambda s: mapper.get(s) if s in mapper else s)
    return data

def cleanYCol(yCol):
    yCol = yCol.astype(int) 
    return yCol

df = pd.read_csv("/Users/hankgansert/Desktop/ML/MachineLearning/MLProject/train_final.csv", header=None)
test = pd.read_csv("/Users/hankgansert/Desktop/ML/MachineLearning/MLProject/test_final.csv", header=None)

targetColumn = 'income>50K'

#Restructure training data 
df.columns = df.iloc[0]
df = df.drop(df.index[0])
df = df.reset_index()

#Restructure testing data 
test.columns = test.iloc[0]
test = test.drop(test.index[0])
xTest = test.reset_index()

#Create X and Y dataframes for training and test
yTrain = df[targetColumn]
yTrain.columns = [targetColumn]





xTrain = df.drop(targetColumn, axis=1)

print(xTrain.dtypes)

xTrain = cleanStringValues(xTrain)
yTrain = cleanYCol(yTrain)
xTest = cleanStringValuesTest(xTest)

DecisionClassifier = tree.DecisionTreeClassifier()

predictionTree = DecisionClassifier.fit(xTrain, yTrain)

results = predictionTree.predict(xTest)


f = open('results.csv', 'w')
writer = csv.writer(f)
header = ['ID','Prediction']
writer.writerow(header)
for i in range(len(results)):
    row = [str(i+1), results[i]]
    writer.writerow(row)







print('poop')





