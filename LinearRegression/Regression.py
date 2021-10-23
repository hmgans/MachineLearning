# This file uses ML decision tree techiniques
# Author: Hank Gansert
# Date: 09/11/21
import math
import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame



def gradientDecentMethod(df, r, t_limit):

    
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


    wFinal, costs, iteration = find_linear_regression_batch(data, w, r, 0, t_limit, 'col8')


    finalCost = costFunction(wFinal,data,'col8')


    x, y = iteration, costs
    plt.plot(x, y)
    # naming the x axis
    plt.xlabel('Iteration')
    # naming the y axis
    plt.ylabel('Error')

    plt.title('Convergence for Batch')

    plt.show()

    print('Final Costs: ' + str(finalCost))
    print('Iteration of Convergence: ' + str(iteration[len(iteration)-1]))
    print('R value: ' + str(r))
    print('Final w: ' + str(wFinal))

    return wFinal

def find_linear_regression_batch(df, w, r, t, t_limit, target):
    costs = []
    iteration = []


    while(t != t_limit):

        cost = costFunction(w, df, target)
        costs.append(cost)
        iteration.append(t)
    
        gradients = []
        for key in df.keys():
            if key == target:
                break
            # Get gradients for all columns
            gradients.append(gradient(w,df,target,key))
    
        multiplicative = r *np.array(gradients)

        temp = np.array(w) - multiplicative
        if hasConverged(temp, w):
            return temp, costs, iteration
        w=temp

        t += 1
    return w, costs, iteration



def find_linear_regression_stoch(df, w, r, t, t_limit, target):
    costs = []
    iteration = []

    while(t != t_limit):

        cost = costFunction(w, df, target)
        costs.append(cost)
        iteration.append(t)



        #Randomly sample a training example
        choice = random.randint(0, len(df)-1) # Get random index
        example = df.iloc[choice]
        #example = df.iloc[0]
        temp = []

        cost = costFunction(w, df, target)

        for i in range(len(w)):
            temp.append(w[i])



        for i in range(len(w)):
            temp[i] = w[i] + r*(getResult(w, example,df.keys()[i], target))

        if hasConverged(temp, w):
            return temp, costs, iteration
        w = temp
        t += 1

    return w, costs, iteration



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
    plt.plot(x, y)
    # naming the x axis
    plt.xlabel('Iteration')
    # naming the y axis
    plt.ylabel('Error')

    plt.title('Convergence for Stochastic')

    plt.show()

    print('Final Costs: ' + str(finalCost))
    print('Iteration of Convergence: ' + str(iteration[len(iteration)-1]))
    print('R value: ' + str(r))
    print('Final w: ' + str(wFinal))



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


def hasConverged(w, prevW):

    result = np.linalg.norm(np.array(w)-np.array(prevW))

    if result < .00001:
        return True
    return False
    

def analyticalResult(df):
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

    Y = pd.DataFrame(data['col8'])
    X = pd.DataFrame(data.drop('col8', axis=1))
    Y = Y.to_numpy()
    X = X.to_numpy()
    X = X.T
    # Perform the Analytical Calculation
    w = np.linalg.inv((np.dot(X,X.T)))
    x = np.dot(w,X)
    y = np.dot(x,Y)




    return y

def costFunctionBeforeCleaning(w, df):
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
    total = 0

    for index, row in df.iterrows():
        
        keys = df.keys()
        estimate = 0
        # Calculate wTx
        for j in range(len(df.keys())-1):
            estimate += row[keys[j]]*w[j]
        #(y - est)
        result = row['col8'] - estimate
        total += math.pow(result,2)
    return total/2


df = pd.read_csv("concrete/train.csv", header=None)

test = pd.read_csv("concrete/train.csv")


gradientW = gradientDecentMethod(df, .0078125, 500)

stochasticW = stochasticGradientDescent(df, .0078125, 10000)

trueResult = analyticalResult(df)

print("Real Result")
print(trueResult)

print("Cost for Batch")
print(costFunctionBeforeCleaning(gradientW, test))

print("Cost for Stochastic")
print(costFunctionBeforeCleaning(stochasticW, test))

print("Cost for Real Result")
print(costFunctionBeforeCleaning(trueResult, test))
