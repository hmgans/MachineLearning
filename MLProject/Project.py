import math
import pandas as pd
import numpy as np
from random import randrange
import random
import sklearn.ensemble as en
import sklearn.tree as tr
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from sklearn.preprocessing import StandardScaler

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

    columnsWithUnknown = []
    for key in df.keys():
        for val in df[key].unique():
            if val == "?":
                columnsWithUnknown.append(key)
    
    for col in columnsWithUnknown:
        attr = get_majority_for_unknown(df, col)
        df[col] = df[col].replace(['?'], attr)
    
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

# This function converts numerical values to + or - based on the median value 
def convertNumericalValues(df):
    medians = df.median()
    indexNames = medians.index.values
    for name in indexNames:
        df[name] = df[name].apply(lambda x: '+' if int(x) > medians[name] else '-')

    return df



def DataFrameCleaningScalar(df, test):
    medians = df.median()
    numericalColumns = medians.index.values
    categoricalColumns = []
    for value in df.columns:
        if value not in numericalColumns:
            categoricalColumns.append(value)

    hotOnesTrain = []
    hotOnesTest = []

    for category in categoricalColumns:
        one_hot_train = pd.get_dummies(df[category])
        one_hot_test = pd.get_dummies(test[category])
        hotOnesTrain.append(one_hot_train)
        hotOnesTest.append(one_hot_test)
        df = df.drop(category,axis = 1)
        test = test.drop(category,axis = 1)

    



    scalar = StandardScaler()
    scalar.fit(df)
    df[df.columns] = scalar.transform(df[df.columns])
    test[test.columns] = scalar.transform(test[test.columns])
    for cat in hotOnesTrain:
        df = df.join(cat)

    for cat in hotOnesTest:
        test = test.join(cat)
    

    

    



    return df, test


def DataFrameCleaning(df, test):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df)

    df = enc.transform(df).toarray()
    test = enc.transform(test)


    return df, test

df = pd.read_csv("/Users/hankgansert/Desktop/MachineLearning/MLProject/train_final.csv", header=None)
test = pd.read_csv("/Users/hankgansert/Desktop/MachineLearning/MLProject/test_final.csv", header=None)

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
BackPropNN = MLPClassifier()
SVM = svm.SVC()


xTrain = df.drop(targetColumn, axis=1)





xTrain = cleanTrain(xTrain)
xTest = cleanTest(xTest)
xTrain = cleanUnknownValues(xTrain)
xTest = cleanUnknownValues(xTest)



yTrain = cleanYCol(yTrain)


#Change numerical values

# xTrain = convertNumericalValues(xTrain)
# xTest = convertNumericalValues(xTest)



xTrain, xTest = DataFrameCleaningScalar(xTrain, xTest)



extra = 0
name = ''
for cat in xTest.columns:
    if cat not in xTrain.columns:
        name = ''
        extra = xTest[cat]
        xTest = xTest.drop(cat, axis=1)

xTest[name] = extra
xTrain[name] = 0



#Ensemble
#predictionTree = DecisionClassifier.fit(xTrain, yTrain)
predictionTree = AdaBoost.fit(xTrain, yTrain)
#predictionTree = RandomForest.fit(xTrain, yTrain)

#NN
#predictionTree = BackPropNN.fit(xTrain, yTrain)

#SVM
#predictionTree = SVM.fit(xTrain, yTrain)


        


        


results = predictionTree.predict(xTest)




f = open('results.csv', 'w')
writer = csv.writer(f)
header = ['ID','Prediction']
writer.writerow(header)
for i in range(len(results)):
    row = [str(i+1), results[i]]
    writer.writerow(row)












