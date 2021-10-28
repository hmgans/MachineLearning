import math
import pandas as pd
import numpy as np
from random import randrange
import random
import sklearn.ensemble as en
import sklearn.tree as tr
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import csv

def get_majority_for_unknown(df, labelName):

    variables = df[labelName].unique()
    greatestAttr = ''
    greatestSize = -1

    for variable in variables:

        if len(df[labelName][df[labelName]==variable]) > greatestSize and variable!='unknown':
            greatestSize = len(df[labelName][df[labelName]==variable])
            greatestAttr = variable

    return greatestAttr

def cleanUnknownValues(df):
    length = df.loc[0].size
    labels = []


    for i in range(length):
        labels.append("col" + str(i))

    df.columns = labels



    columnsWithUnknown = []
    for key in df.keys():
        for val in df[key].unique():
            if val == "unknown":
                columnsWithUnknown.append(key)
    
    for col in columnsWithUnknown:
        attr = get_majority_for_unknown(df, col)
        df[col] = df[col].replace(['unknown'], attr)
    
    return df



# This function will map string values to a unique number
def cleanTrain(data):
    data = data.drop(['index'],axis=1)
    return data

def cleanTest(data):

    data = data.drop(['ID'],axis=1)
    data = data.drop(['index'],axis=1)
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


DecisionClassifier = tr.DecisionTreeClassifier()
AdaBoost = en.AdaBoostClassifier()
RandomForest = en.RandomForestClassifier()


xTrain = df.drop(targetColumn, axis=1)





def DataFrameCleaning(df, test):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df)

    df = enc.transform(df).toarray()
    test = enc.transform(test)


    return df, test

xTrain = cleanTrain(xTrain)
xTest = cleanTest(xTest)
xTrain = cleanUnknownValues(xTrain)
xTest = cleanUnknownValues(xTest)



yTrain = cleanYCol(yTrain)




xTrain, xTest = DataFrameCleaning(xTrain, xTest)





#predictionTree = DecisionClassifier.fit(xTrain, yTrain)
predictionTree = AdaBoost.fit(xTrain, yTrain)
#predictionTree = RandomForest.fit(xTrain, yTrain)

results = predictionTree.predict(xTest)




f = open('results.csv', 'w')
writer = csv.writer(f)
header = ['ID','Prediction']
writer.writerow(header)
for i in range(len(results)):
    row = [str(i+1), results[i]]
    writer.writerow(row)







print('poop')





