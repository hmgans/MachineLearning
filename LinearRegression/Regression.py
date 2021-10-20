# This file uses ML decision tree techiniques
# Author: Hank Gansert
# Date: 09/11/21
import math
import pandas as pd
import numpy as np





def gradientDecentMethod(df, targetColumn, r, t_limit):

    length = df.loc[0].size
    labels = []
    iter = 0
    treeCount = {}
    trees = {}
    # Add a collumn of ones for the b value

    columnOfOnes = []
    for i in range(len(df)):
        columnOfOnes.append(1)
        
    df['colOnes'] = columnOfOnes




    for i in range(length):
        labels.append("col" + str(i))

    df.columns = labels



    newCol = df[targetColumn]

    data = df.drop(targetColumn, axis=1)



    data[targetColumn] = newCol

    data.columns = labels

    w = []
    for i in range(length-1):
        w.append(0)

    find_linear_regression(df, w, r, 0, t_limit, targetColumn)




    return 0

def find_linear_regression(df, w, r, t, t_limit, target):

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
    return find_linear_regression(df, w, r, t+1,t_limit, target)



    

    

    return 0

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


def stochasticGradientDescent(df, r):

    return 0



df = pd.read_csv("/Users/hankgansert/Desktop/ML/MachineLearning/LinearRegression/concrete/train.csv", header=None)


gradientDecentMethod(df, 'col7', .1, 10)
