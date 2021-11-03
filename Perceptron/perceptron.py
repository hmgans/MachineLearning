# This file uses ML decision tree techiniques
# Author: Hank Gansert
# Date: 09/11/21
import math
from os import error
import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame


def getResult(w, row, key, target):
    keys = df.keys()
    result = 0
    for j in range(len(df.keys())-1):
         result += row[keys[j]]*w[j]
        #(y - est)
    result = row[target] - result
    result *= row[key]
    return result

def batch(w, df):

    return 0

def gradient(w, df, target, column):
    total = 0
    for index, row in df.iterrows():
        
        keys = df.keys()
        estimate = 0
        # Calculate wTx
        for j in range(len(df.keys())-1):
            estimate += row[keys[j]]*w[j]
        #(y - est)
        result = row[target] - estimate
        result = result*row[column]
        total += result
    
    return -total


def stochasticGradientDescent(df, r, t_limit):
    labels = []
    iter = 0
    treeCount = {}
    trees = {}
    switchColumn = 'col7'
    # Add a collumn of ones for the b value

    columnOfOnes = []
    for i in range(len(df)):
        columnOfOnes.append(1)
        
    # Column of ones for the B value

    df['colOnes'] = columnOfOnes
    length = df.loc[0].size




    for i in range(length):
        labels.append("col" + str(i))

    df.columns = labels



    newCol = df[switchColumn]

    data = df.drop(switchColumn, axis=1)



    data[switchColumn] = newCol

    data.columns = labels

    w = []
    for i in range(length-1):
        w.append(0)

    wFinal, costs, iteration = find_linear_regression_stoch(data, w, r, 0, t_limit, 'col8')

    finalCost = costFunction(wFinal,data,'col8')


    x, y = iteration, costs



    return wFinal

def costFunction(w, df, target):

    total = 0
    for index, row in df.iterrows():
        
        keys = df.keys()
        estimate = 0
        # Calculate wTx
        for j in range(len(df.keys())-1):
            estimate += row[keys[j]]*w[j]
        #(y - est)
        result = row[target] - estimate
        total += math.pow(result,2)
    return total/2






#Find the prediction for the row
def predict(w, row, df):
    keys = df.keys()

    predicition = 0
    for i in range(len(w)):
        predicition += w[i]*row[keys[i]]
    if predicition >= 0:
        return 1.0
    else:
        return 0.0

# This function clleans the dataframe and adds a column of 1's to represent b
# Note - target is second to last column.
def cleanDataFrame(df, target):
    labels = []
    switchColumn = 'col' +str(target-1)
    # Add a collumn of ones for the b value


    # Column of ones for the B value

    length = df.loc[0].size




    for i in range(length):
        labels.append("col" + str(i))

    df.columns = labels



    newCol = df[switchColumn]

    data = df.drop(switchColumn, axis=1)



    data[switchColumn] = newCol

    data.columns = labels

    return data

# This is the regular perceptron 
# df - cleaned dataframme
# epoch - number of epochs
# r - step size
def perceptronReg(df, epoch, r):
    # Create a w vector of zeros 
    length = df.loc[0].size
    w = []
    for i in range(length-1):
        w.append(0)

    w = np.array(w)

    for e in range(epoch):
        # Randomly Shuffle
        df = df.sample(frac=1).reset_index(drop=True)
        for index, row in df.iterrows():
            y = row[-1]
            if y == 0:
                y = -1
            
            x = np.array(row[:-1])

            result = np.dot(w.T,x)
            result *= y

            # If there is an error update
            if result <= 0:
                w = w + r * y * x
 
    return w

#Finds the 
def averageErrorReg(w, df):
    target = df.keys()[-1]
    errortotal = 0
    total =0
    for index, row in df.iterrows():
        total += 1
        y = row[-1]
        if y == 0:
            y = -1
        x = np.array(row[:-1])

        result = np.dot(x,w)
        result *= y
        
        if result <= 0:
            errortotal += 1
  
    return errortotal/total

# This is the regular perceptron 
# df - cleaned dataframme
# epoch - number of epochs
# r - step size
# return tuple of vectors and counts
def perceptronVoted(df, epoch, r):
    # Create a w vector of zeros 
    length = df.loc[0].size
    w = []
    for i in range(length-1):
        w.append(0)

    w = np.array(w)

    predictors = []

    for e in range(epoch):
        # No Shuffle
        # Set initial count
        c = 0
        for index, row in df.iterrows():
            y = row[-1]
            if y == 0:
                y = -1
            
            x = np.array(row[:-1])


            result = np.dot(w.T,x)
            result *= y

            # If there is an error update
            if result <= 0:
                #Save count and the weight vector
                item = (w,c)
                predictors.append(item)

                w = w + r * y * x
                #Initial count for the new w
                c = 0
            else:
                # if correct, add to count
                c += 1
 
    return predictors

#Finds the 
def averageErrorVote(w, df):
    target = df.keys()[-1]
    errortotal = 0
    total =0
    for index, row in df.iterrows():
        total += 1
        y = row[-1]
        if y == 0:
            y = -1
        x = np.array(row[:-1])

        totalResult = 0
        for i in range(len(w)):
            result = np.dot(x,w[i][0])
            if result <= 0:
                result = -1
            else:
                result = 1
            totalResult += w[i][1]*result

        result *= y
        
        if result <= 0:
            errortotal += 1
  
    return errortotal/total

# This is the regular perceptron 
# df - cleaned dataframme
# epoch - number of epochs
# r - step size
# returns weight vector
def perceptronAverage(df, epoch, r):
    # Create a w vector of zeros 
    length = df.loc[0].size
    a = []
    w = []
    for i in range(length-1):
        w.append(0)

    #Create 'a' vector
    for i in range(length-1):
        a.append(0)

    w = np.array(w)
    a = np.array(a)

    # c keeps track of the total iterations to calculate average
    c = 0

    for e in range(epoch):
        # No Shuffle
        # Set initial count
        
        for index, row in df.iterrows():

            c += 1
            y = row[-1]
            if y == 0:
                y = -1
            
            x = np.array(row[:-1])


            result = np.dot(w.T,x)
            result *= y

            # If there is an error update
            if result <= 0:
                w = w + r * y * x

            a = a + w

    return a/c

df = pd.read_csv("bank-note/train.csv", header=None)
test = pd.read_csv("bank-note/test.csv")

#Clean data
data = cleanDataFrame(df, 5)
testData = cleanDataFrame(test, 5)


wReg = perceptronReg(data,10,.1)
errorReg = averageErrorReg(wReg,testData)
print('This is the Regular Perceptron Weight Vector\\\\')
print(str(wReg) + "\\\\")
print('Error: ' + str(errorReg)+"\\\\\\\\")



wVote = perceptronVoted(data,10,.1)
for i in range(len(wVote)):
    if(wVote[i][1]!=0):
        print("This is the weight vector at "+str(i)+": $" + str(wVote[i][0]) + str('$\\\\'))
        print("Count: "+ str(wVote[i][1])+ str('\\\\'))
errorVote = averageErrorVote(wVote, testData)
print('Average Error Voted Perceptron: '+ str(errorVote)+"\\\\")

wAvg = perceptronAverage(df, 10, .1)
errorAvg = averageErrorReg(wAvg,testData)
print('This is the Average Weight Vector\\\\')
print(str(wAvg)+"\\\\")
print('Error: ' + str(errorAvg)+"\\\\")




