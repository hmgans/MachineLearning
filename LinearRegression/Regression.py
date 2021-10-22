# This file uses ML decision tree techiniques
# Author: Hank Gansert
# Date: 09/11/21
import math
import pandas as pd
import numpy as np
import random





def gradientDecentMethod(df, r, t_limit):

    
    labels = []
    iter = 0
    treeCount = {}
    trees = {}
    switchColumn = 'col3'
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

    w = [-1,1,-1,-1]

    find_linear_regression_batch(data, w, r, 0, t_limit, 'col4')




    return 0

def find_linear_regression_batch(df, w, r, t, t_limit, target):

    if(t == t_limit):
        return w
    
    gradients = []
    for key in df.keys():
        if key == target:
            break
        # Get gradients for all columns
        gradients.append(gradient(w,df,target,key))
    
    multiplicative = r *np.array(gradients)
    w = np.array(w) - multiplicative
    return find_linear_regression_batch(df, w, r, t+1,t_limit, target)



def find_linear_regression_stoch(df, w, r, t, t_limit, target):

    if(t == t_limit):
        return w

    #Randomly sample a training example
    choice = random.randint(0, len(df)-1) # Get random index
    example = df.iloc[choice]
    #example = df.iloc[0]
    temp = []

    cost = costFunction(w, df, target)
    print(str(cost))

    for i in range(len(w)):
        temp.append(w[i])



    for i in range(len(w)):
        temp[i] = w[i] + r*(getResult(w, example,df.keys()[i], target))

    if hasConverged(temp, w):
        return temp
    w = temp
    



    

    return find_linear_regression_batch(df, w, r, t+1,t_limit, target)



def getResult(w, row, key, target):
    result = 0
    for j in range(len(df.keys())-1):
         result += row[key]*w[j]
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

        




    return 0


def stochasticGradientDescent(df, r, t_limit):
    labels = []
    iter = 0
    treeCount = {}
    trees = {}
    switchColumn = 'col3'
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

    find_linear_regression_stoch(data, w, r, 0, t_limit, 'col4')



    return 0

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

def hasConverged(w, prevW):

    result = np.linalg(w-prevW)

    if result < .00001:
        return True
    return False
    


df = pd.read_csv("/Users/hankgansert/Desktop/ML/MachineLearning/LinearRegression/tester.csv", header=None)


gradientDecentMethod(df, .1, 100)

stochasticGradientDescent(df, .1, 100)
