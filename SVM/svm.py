# This file uses ML SVM
# Author: Hank Gansert
# Date: 09/11/21
import math
from os import error
from numpy.lib.index_tricks import CClass
import pandas as pd
import numpy as np
import random
from pandas.core.construction import array
import scipy
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame




#This function with replac unknown values in the dataframe with the majority 
def clean(df):

    length = df.loc[0].size
    df[4] = df[4].apply(lambda x: -1 if x==0 else 1 )
    df.insert(loc=4,column='4',value=1)
    
    return df






# This is the regular perceptron 
# df - cleaned dataframme
# epoch - number of epochs
# r - step size
def SVM_TEST(df, epoch, r, C):
    # Create a w vector of zeros 
    length = df.loc[0].size
    w = []
    for i in range(length-1):
        w.append(0)

    w = np.array(w)
    N = length - 1
    rs = [.01,.005,.0025]


    for e in range(epoch):
        # Randomly Shuffle
        #df = df.sample(frac=1).reset_index(drop=True)
        for index, row in df.iterrows():
            r = rs[index]
            row = np.array(row)
            y = row[-1]
            
            x = np.array(row[:-1])

            result = np.dot(w.T,x)
            result *= y

            # Small Change Here for the SVM figure out what N is and what C should be and [w;0]

            if result <= 1:
                w = (1-r)*w[:-1] + r * C * N * x
            else:
                w = (1-r)*w[:-1]
            
    return w


# This is the regular perceptron 
# df - cleaned dataframme
# epoch - number of epochs
# r - step size
def SVM_SG(df, epoch, r, C, a, problem):
    # Create a w vector of zeros 
    length = df.loc[0].size

    w = []
    for i in range(length-1):
        w.append(0)

    w = np.array(w)
    N = len(df)
    weights = []

    for e in range(epoch):

        # Save weights for each epoch
        weights.append(w)
        if problem == 'a':
            rt = r/(1+(math.pow(r,e)/a))
        elif problem == 'b':
            rt = r/(1+e)
        else:
            rt = r

        # Randomly Shuffle
        df = df.sample(frac=1).reset_index(drop=True)
        for index, row in df.iterrows():
            row = np.array(row)
            y = row[-1]

            
            x = np.array(row[:-1])

            result = np.dot(w.T,x)

            result *= y

            # Small Change Here for the SVM figure out what N is and what C should be and [w;0]
            if result <= 1:
                temp = ((1-rt)*w[:-1]).tolist()
                temp.append(0)
                w = np.array(temp) + rt * C * N * y * x
            else:
                temp = ((1-rt)*w[:-1]).tolist()
                temp.append(0)
                w = np.array(temp)

            
    return weights

#Finds the average error
def averageErrorReg(w, df):
    target = df.keys()[-1]
    errortotal = 0
    total = 0
    for index, row in df.iterrows():
        row = np.array(row)
        total += 1
        y = row[-1]
        x = np.array(row[:-1])

        result = np.dot(x.T,w)
        result *= y
        
        if result <= 0:
            errortotal += 1
  
    return errortotal/total

        

def objective(a, x, y):


    yNbyN = np.outer(np.array(y),np.transpose(np.array(y)))
    aNbyN = np.outer(np.array(a),np.transpose(np.array(a)))
    xNbyN = np.asmatrix(x)
    xNbyN = xNbyN * np.transpose(xNbyN)

    result = .5 * np.sum(yNbyN * aNbyN * xNbyN) - np.sum(a)


    return result



def objectiveTrun(a, x, y):

    result = 0


    for i in range(len(x)):
        for j in range(len(x)):
            result+= y[i]*y[j]*a[i]*a[j]*np.dot(x[i].T, x[j])

    # Got inner result so then multiply by half
    result *= .5

    subtractAlphas = 0
    for i in range(len(a)):
        subtractAlphas += a[i]

    result = result - subtractAlphas


    return result
    











def SVM_DUAL(df, C):

    bnds = []
    # alpha bounds to [0,C]
    for i in range(len(df)):
        # Is this inclusive or exclusive
        bnds.append((0,C))

    a = []
    #initialze alphas to 0
    for i in range(len(df)):
        a.append(random.uniform(0, C))
    

    yArr = []
    xArr = []
    #Structure Data
    for index, row in df.iterrows():
        row = np.array(row)
        y = row[-1]

        x = np.array(row[:-1])

        xArr.append(x)
        yArr.append(y)

    

    cons = {'type':'eq', 'fun': lambda x: np.sum(np.array(a)*np.array(yArr))}
    
    
    result = minimize(objective,x0=a, args=(xArr,yArr) ,method='SLSQP',constraints=cons,bounds=bnds)


    w = [0,0,0,0,0]
    for i in range(len(xArr)):
        w += result.x[i]*yArr[i]* xArr[i]

    return w

def Problem1A():
    df = pd.read_csv("/Users/hankgansert/Desktop/Temp/ML/MachineLearning/SVM/bank-note/train.csv", header=None)
    test = pd.read_csv("/Users/hankgansert/Desktop/Temp/ML/MachineLearning/SVM/bank-note/test.csv", header=None)
    df = clean(df)
    test = clean(test)
    C = [100/873,500/873,700/873]
    for i in range(len(C)):
        w = SVM_SG(df, 100, .001, C[i], 1, 'a')
        epoch = []
        errorTrain = []
        errorTest = []
        fig, ax = plt.subplots()
        for j in range(len(w)):
            epoch.append(j)
            errorTrain.append(averageErrorReg(w[j],df))
            errorTest.append(averageErrorReg(w[j],test))
        ax.plot(epoch,errorTrain)
        ax.plot(epoch,errorTest)
        # naming the x axis
        ax.set_xlabel('Epoch')
        # naming the y axis
        ax.set_ylabel('Error')
 
        # giving a title to my graph
        ax.set_title('Epoch Error Rate for when C='+str(C[i]))
        plt.show()


def Problem1B():
    df = pd.read_csv("/Users/hankgansert/Desktop/Temp/ML/MachineLearning/SVM/bank-note/train.csv", header=None)
    test = pd.read_csv("/Users/hankgansert/Desktop/Temp/ML/MachineLearning/SVM/bank-note/test.csv", header=None)
    df = clean(df)
    test = clean(test)

    C = [100/873,500/873,700/873]
    for i in range(len(C)):
        w = SVM_SG(df, 100, .001, C[i], 1, 'b')
        epoch = []
        errorTrain = []
        errorTest = []
        fig, ax = plt.subplots()
        for j in range(len(w)):
            epoch.append(j)
            errorTrain.append(averageErrorReg(w[j],df))
            errorTest.append(averageErrorReg(w[j],test))
        ax.plot(epoch,errorTrain)
        ax.plot(epoch,errorTest)
        # naming the x axis
        ax.set_xlabel('Epoch')
        # naming the y axis
        ax.set_ylabel('Error')
 
        # giving a title to my graph
        ax.set_title('Epoch Error Rate for when C='+str(C[i]))
        plt.show()

#df = pd.read_csv("bank-note/train.csv", header=None)
#test = pd.read_csv("bank-note/test.csv")

#This was for debugging and completing problem 5.
#df = pd.read_csv("/Users/hankgansert/Desktop/Temp/ML/MachineLearning/SVM/tester.csv", header=None)
#SVM_TEST(df, 3, 0, 1/3)

# Program Part 1.A
Problem1A()

# Program Part 1.B
#Problem1B()

df = pd.read_csv("/Users/hankgansert/Desktop/Temp/ML/MachineLearning/SVM/bank-note/train.csv", header=None)
df = clean(df)
#Program Part 2
w = SVM_DUAL(df, 1/2)
print(averageErrorReg(w,df))

        




