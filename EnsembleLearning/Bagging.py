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



smallNumber = .000001
def get_majority_for_unknown(df, labelName):

    variables = df[labelName].unique()
    greatestAttr = ''
    greatestSize = -1

    for variable in variables:

        if len(df[labelName][df[labelName]==variable]) > greatestSize and variable!='unknown':
            greatestSize = len(df[labelName][df[labelName]==variable])
            greatestAttr = variable

    return greatestAttr


def get_majority(df, labelName):

    variables = df[labelName].unique()
    greatestAttr = ''
    greatestSize = -1

    for variable in variables:

        if len(df[labelName][df[labelName]==variable]) > greatestSize:
            greatestSize = len(df[labelName][df[labelName]==variable])
            greatestAttr = variable

    return greatestAttr

def get_majority_ada(df, labelName):

    variables = df[labelName].unique()
    greatestAttr = ''
    greatestSize = -1

    for variable in variables:

        if df['weights'][df[labelName]==variable].sum() > greatestSize:
            greatestSize = df['weights'][df[labelName]==variable].sum()
            greatestAttr = variable

    return greatestAttr


def get_total_entropy(df, labelName):

    entropy = 0


    labels = df[labelName].unique() 

    for label in labels:
        fraction = df[labelName].value_counts()[label]/len(df[labelName])
        entropy += -fraction*np.log2(fraction)
    return entropy
  
  
def get_total_GI(df, labelName):

    totalGini = 0

    labels = df[labelName].unique() 

    for label in labels:
        fraction = df[labelName].value_counts()[label]/len(df[labelName])
        totalGini += math.pow(fraction,2)
    return 1- totalGini

def get_total_ME(df, labelName):


    labels = df[labelName].unique() 
    greatest = -1
    greatestLabel = 'none'
    for label in labels:

        if df[labelName].value_counts()[label] > greatest:
            greatestLabel = label
        
    fraction = df[labelName].value_counts()[greatestLabel]
    return fraction



def find_entropy_attribute(df, attribute, labelName):

  target_variables = df[labelName].unique()  #This gives all 'Yes' and 'No'
  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
  entropy2 = 0

  # Get each label for the specific attribute.
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:

        # Gets the number of attributes with a specific value that also has the target value in the target column
          num = len(df[attribute][df[attribute]==variable][df[labelName] ==target_variable])

          # Gets the number of of attributes with specific value 
          den = len(df[attribute][df[attribute]==variable])

          # Creates fraction for computation
          fraction = num/(den+smallNumber)

          # Calculates Entropy for that specific target variable example
          entropy += -fraction*math.log(fraction+smallNumber)

      # Multiply the the entropy result by the fraction of labels with attribute / total rows
      fraction2 = den/len(df)

      # Add all variables entropy togther to get accumulated result.
      entropy2 += -fraction2*entropy
  return abs(entropy2)

def find_GI_attribute(df, attribute, labelName):

  target_variables = df[labelName].unique()  #This gives all 'Yes' and 'No'
  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
  totalginindex = 0

  # Get each label for the specific attribute.
  for variable in variables:
      giniIndex = 0
      for target_variable in target_variables:

        # Gets the number of attributes with a specific value that also has the target value in the target column
          num = len(df[attribute][df[attribute]==variable][df[labelName] ==target_variable])

          # Gets the number of of attributes with specific value 
          den = len(df[attribute][df[attribute]==variable])

          # Creates fraction for computation
          fraction = num/(den+smallNumber)

          # Calculates Entropy for that specific target variable example
          giniIndex += math.pow(fraction,2)

      # Multiply the the entropy result by the fraction of labels with attribute / total rows
      fraction2 = den/len(df)

      # Add all variables entropy togther to get accumulated result.
      totalginindex += fraction2*(1-giniIndex)
  return abs(totalginindex)

def find_ME_attribute(df, attribute, labelName):

  target_variables = df[labelName].unique()  #This gives all 'Yes' and 'No'
  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
  majorityError = 0

  # Get each label for the specific attribute.
  for variable in variables:
      greatest = -1
      greatestLabel = 'none'

      for target_variable in target_variables:

        # Gets the number of attributes with a specific value that also has the target value in the target column
          if greatest < len(df[attribute][df[attribute]==variable][df[labelName] ==target_variable]):
              greatest = len(df[attribute][df[attribute]==variable][df[labelName] ==target_variable])

          # Gets the number of of attributes with specific value 
          den = len(df[attribute][df[attribute]==variable])

          

      fraction = (den - greatest)/den
      fraction2 = fraction*(den/len(df))

      # Add all variables entropy togther to get accumulated result.
      majorityError += fraction2
  return abs(majorityError)


def get_highest_IG(df, labelName, gainType):
    Entropy_att = []
    IG = []
    labelArray = np.array([labelName])
    keys = df.keys().to_numpy()
    result = np.setdiff1d(keys, labelArray)
    
    for key in df.keys()[:-1]:
        
        
        # Information gain for each attribute and return the key with the highest gain. 
        if gainType == 'ME':
            IG.append(get_total_ME(df, labelName)-find_ME_attribute(df,key,labelName))
        elif gainType == 'GI':
            IG.append(get_total_GI(df, labelName)-find_GI_attribute(df,key,labelName))
        else:
            IG.append(get_total_entropy(df, labelName)-find_entropy_attribute(df,key,labelName))

    return df.keys()[:-1][np.argmax(IG)]

    

def get_highest_IG_RandomForest(df, labelName, gainType):

    # Create a featureSubset
    options = df.keys()[:-1]
    # Get either 2,4,6
    opt = [2,4,6]
    random_index = random.randint(0,len(opt)-1)
    size = opt[random_index]
    #Save the new Keys
    theNewKeys = []
    if size > len(options):
        options = theNewKeys
    else:
        for i in range(size):
            randomColumnIndex = random.randint(0,len(options)-1)
            # Prevent same thing from being chosen twice, also make sure that the size is good.
            selection = options[randomColumnIndex]

            theNewKeys.append(selection)
            options.delete(randomColumnIndex)



    
    IG = []
    
    for key in theNewKeys:
        
        
        # Information gain for each attribute and return the key with the highest gain. 
        if gainType == 'ME':
            IG.append(get_total_ME(df, labelName)-find_ME_attribute(df,key,labelName))
        elif gainType == 'GI':
            IG.append(get_total_GI(df, labelName)-find_GI_attribute(df,key,labelName))
        else:
            IG.append(get_total_entropy(df, labelName)-find_entropy_attribute(df,key,labelName))

    return theNewKeys[np.argmax(IG)]
  


  
def get_subtable(df, node,value):
  return df[df[node] == value].reset_index(drop=True)

# Builds a decision Tree
# df - dataframe to use
# labelName - label to use or column, 'col1', 'col2'
# height - input as 0 to start
# gainType - "ME" - Majority Error, "IG" -information gain, or "GI" - gini index
def buildTree(df, labelName, height, limit, gainType, tree=None): 
    
    newNode = get_highest_IG(df, labelName, gainType)

    labelValues = np.unique(df[newNode])





    
    if tree is None:                    
        tree={}
        tree[newNode] = {}
    
    #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 

    for value in labelValues:
        
        nodetable = get_subtable(df, newNode, value)
        

        clValue, counts = np.unique( nodetable[labelName] , return_counts=True) 

        size = len(clValue)   
        getSame = get_all_same(nodetable, labelName, gainType)

        #Check if there is a repeat on the Node, then we know there is no best answer and we need to take the majority.
        majority = False
        if 'allsame' == getSame and height < limit:
            buildTree(nodetable, labelName, height+1, limit, gainType)
        elif 'allsame' == getSame:
            majority = True

            
            
        
                
        # When count length is one we know there is only one label for that subtable in that node 
        if len(counts)==1 or height==limit or majority: 

            if len(counts)==1:
                tree[newNode][value] = clValue[0]  
            else:
                tree[newNode][value] =  get_majority(df, labelName)
                                                          
        else:        
            tree[newNode][value] = buildTree(nodetable, labelName, height+1, limit, gainType) #Calling the function recursively 
                   

    return tree



# Constructs a tree
# df - dataframe to use
# targetColumn - column you want to predict "col1", "col2", etc.
def constructTree(df, targetColumn, limit, gainType):
    length = df.loc[0].size
    labels = []


    for i in range(length):
        labels.append("col" + str(i))

    df.columns = labels



    newCol = df[targetColumn]


    data = df.drop(targetColumn, axis=1)
    data[targetColumn] = newCol



    data.columns = labels


    tree = buildTree(data, data.keys()[-1], 0, limit, gainType)

    return tree

# Used in testing 
def create_test_frame(df, targetColumn):
    length = df.loc[0].size
    labels = []


    for i in range(length):
        labels.append("col" + str(i))

    df.columns = labels



    newCol = df[targetColumn]

    data = df.drop(targetColumn, axis=1)
    data[targetColumn] = newCol
    

    data.columns = labels

    return data

# recusively find the result 
def find_result(inst,tree):

    for nodes in tree.keys():        
        
        value = inst[nodes]
        
        if tree[nodes].get(value) == None:
            prediction = None
            break;
        tree = tree[nodes][value]
        prediction = 0
            
        if type(tree) is dict:
            prediction = find_result(inst, tree)
        else:
            prediction = tree
            break;                            
        
    return prediction





#This function wwill test the data once the decision tree is made
# tree - decision tree
# test - dataframe for test
# label - column you want to test, 'col1', 'col2' etc.
def test_data(tree, test, label):

    
    test = create_test_frame(test, label)
    testCol =test[test.keys()[-1]]
    testCol = pd.DataFrame(testCol)
    test = test.drop(test.keys()[-1], axis=1)

    array = []
    for i in range(len(test)):
        line = pd.Series(test.iloc[i])
        insideArray = []
        insideArray.append(find_result(line, tree))
        array.append(insideArray)

    newCheckArray = pd.DataFrame(np.array(array))

    testCol['results'] = newCheckArray
    labels = ['training', 'results']
    testCol.columns = labels

    testCol['hits'] = testCol.apply( lambda row: row.training == row.results, axis=1)




    total = len(testCol)
    hits = len(testCol['hits'][testCol['hits']==True])
    return hits/total

# This will convert numerical values to a + or - baased on the median.
def cleanNumbericalValues(df):
    length = df.loc[0].size
    labels = []


    for i in range(length):
        labels.append("col" + str(i))

    df.columns = labels

    dataFrame = pd.DataFrame(df.median(axis = 0))
    index = dataFrame.index
    indexList = list(index)

   

    for index in indexList:
        newColumn = []
        for something, row in df.iterrows():
            newColValue = []

            if row[index] > dataFrame.loc[index].values[0]:
                newColValue.append('+')
            else:
                newColValue.append('-')
            newColumn.append(newColValue)
        replacementCol = pd.DataFrame(newColumn)
        df[index] = replacementCol

    return df

#This function with replac unknown values in the dataframe with the majority 
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

def get_all_same_ada(df, labelName, gainType):
    Entropy_att = []
    IG = []
    labelArray = np.array([labelName])
    keys = df.keys().to_numpy()
    result = np.setdiff1d(keys, labelArray)
    
    for key in df.keys()[:-2]:
        IG.append(get_total_entropy_ada(df, labelName)-find_entropy_attribute_ada(df,key,labelName))



    baseValue = IG[0]
    allsame = True
    for value in IG:
        if baseValue != value:
            allsame = False

    if allsame:
        return 'allsame'
    return df.keys()[:-2][np.argmax(IG)]
def get_highest_IG_ada(df, labelName, gainType):
    Entropy_att = []
    IG = []
    labelArray = np.array([labelName])
    keys = df.keys().to_numpy()
    result = np.setdiff1d(keys, labelArray)
    
    for key in df.keys()[:-2]:
        
        
        # Information gain for each attribute and return the key with the highest gain.


        
        IG.append(get_total_entropy_ada(df, labelName)-find_entropy_attribute_ada(df,key,labelName))



    return df.keys()[:-2][np.argmax(IG)]



def find_entropy_attribute_ada(df, attribute, labelName):

  target_variables = df[labelName].unique()  #This gives all 'Yes' and 'No'
  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
  entropy2 = 0

  # Get each label for the specific attribute.
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:

          # Gets the summation of weights for the number of attributes with a specific value that also has the target value in the target column
          num = df['weights'][df[attribute]==variable][df[labelName] ==target_variable].sum()

          # Gets the summation of weights for the number of attributes with specific value 
          den = df['weights'][df[attribute]==variable].sum()

          fraction = num/(den+smallNumber)

          # Calculates Entropy for that specific target variable example
          entropy += -fraction*math.log(fraction+smallNumber)

      # Multiply the the entropy result by the fraction of labels with attribute / total rows
      #
      #fraction2 = den/len(df) summation of everything is 1
      fraction2 = den/1.0

      # Add all variables entropy togther to get accumulated result.
      entropy2 += -fraction2*entropy
  return abs(entropy2)

def get_total_entropy_ada(df, labelName):

    entropy = 0


    labels = df[labelName].unique() 

    for label in labels:
        fraction = df['weights'][df[labelName]==label].sum()
        entropy += -fraction*np.log2(fraction)
    return entropy
   

# Builds a decision Tree
# df - dataframe to use
# labelName - label to use or column, 'col1', 'col2'
# height - input as 0 to start
# gainType - "ME" - Majority Error, "IG" -information gain, or "GI" - gini index
def buildTreeWithAdaBoost(df, labelName, height, limit, gainType, tree=None): 
    
    newNode = get_highest_IG_ada(df, labelName, gainType)

    labelValues = np.unique(df[newNode])





    
    if tree is None:                    
        tree={}
        tree[newNode] = {}
    
    #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 

    for value in labelValues:
        
        nodetable = get_subtable(df, newNode, value)
        

        clValue, counts = np.unique( nodetable[labelName] , return_counts=True) 

        size = len(clValue)   
        getSame = get_all_same_ada(nodetable, labelName, gainType)

        #Check if there is a repeat on the Node, then we know there is no best answer and we need to take the majority.
        majority = False
        if 'allsame' == getSame and height < limit:
            buildTreeWithAdaBoost(nodetable, labelName, height+1, limit, gainType)
        elif 'allsame' == getSame:
            majority = True

            
            
        
                
        # When count length is one we know there is only one label for that subtable in that node 
        if len(counts)==1 or height==limit or majority: 

            if len(counts)==1:
                tree[newNode][value] = clValue[0]  
            else:
                tree[newNode][value] =  get_majority_ada(df, labelName)
                                                          
        else:        
            tree[newNode][value] = buildTreeWithAdaBoost(nodetable, labelName, height+1, limit, gainType) #Calling the function recursively 
                   

    return tree



# Constructs a tree with Ada Boost
# df - dataframe to use
# targetColumn - column you want to predict "col1", "col2", etc.
def constructAdaTree(df, targetColumn, limit, gainType, T):
    length = df.loc[0].size
    labels = []
    iter = 0
    treealpha = []
    trees = []



    for i in range(length):
        labels.append("col" + str(i))

    df.columns = labels



    newCol = df[targetColumn]


    data = df.drop(targetColumn, axis=1)
    data[targetColumn] = newCol



    data.columns = labels

    size = len(df[targetColumn])
    weightCol = np.ones(size)/size

    data['weights'] = weightCol


    tree = buildTreeWithAdaBoost(data, data.keys()[-2], 0, limit, gainType)

    # These three variables are for testing
    totalErrors = 0
    totalErrorArr = []
    currentErrorArr = []

    while iter < T:

        # Next test the tree and and update weights and run again 
        #run tree through the info again and set predicted value column
        # add predicted value column to data
        # find where prediciotn is not the actual result and sum the wieghts.
        resultCol = test_data_ada(tree, data)
        data['results'] = resultCol
        data['realvalue'] = data[targetColumn]

        err = data['weights'][data[targetColumn] != data['results']].sum()


        data['combo'] = data.apply( lambda row: 1 if (row.realvalue == row.results) else -1, axis=1)


        new_weight = np.log((1 - err) / err) / 2

        treealpha.append(new_weight)
        trees.append(tree)



        new_df_weight = (
            data['weights'] * np.exp(-new_weight * data['combo'])
        )
        new_df_weight /= new_df_weight.sum()

        data['weights'] = new_df_weight

        # drop undesired tables
        data = data.drop('results', axis=1)
        data = data.drop('realvalue', axis=1)
        data = data.drop('combo', axis=1)

        tree = buildTreeWithAdaBoost(data, data.keys()[-2], 0, limit, gainType)
        


        iter = iter + 1

    #Now return the final trees and their weights

    return trees, treealpha



def test_data_ada(tree, test):

    
    testCol =test[test.keys()[-2]]
    testCol = pd.DataFrame(testCol)
    newTest = test.drop(test.keys()[-2], axis=1)

    array = []
    for i in range(len(newTest)):
        line = pd.Series(newTest.iloc[i])
        insideArray = []
        insideArray.append(find_result(line, tree)) # contruct a prediction column

        array.append(insideArray)

    newCheckArray = pd.DataFrame(np.array(array))

    
    return newCheckArray

def create_test_frame_ada(df, targetColumn):
    length = df.loc[0].size
    labels = []


    for i in range(length):
        labels.append("col" + str(i))

    df.columns = labels



    newCol = df[targetColumn]

    data = df.drop(targetColumn, axis=1)
    data[targetColumn] = newCol
    

    data.columns = labels

    return data


#Use bootstrap where bagging where m' = m

def subSampleForBagging(dataset):
    # must retrive a list of column names 
    newDataFrame = pd.DataFrame()
    selectables = dataset.keys()[:-1]
    for i in range(len(selectables)):
        choice = random.choice(selectables)

        newDataFrame[choice+'.'+str(i)] = dataset[choice]

    newDataFrame[dataset.keys()[-1]] = dataset[dataset.keys()[-1]]


        

    return newDataFrame

def bootstrapSample(data):
    newDataFrame = pd.DataFrame(columns=data.columns)
    # m' = m
    for i in range(len(data)):

        choice = random.randint(0, len(data)-1) # Get random index
        newDataFrame.loc[len(newDataFrame.index)] = data.iloc[choice]

    return newDataFrame



# Constructs a tree with Ada Boost
# df - dataframe to use
# targetColumn - column you want to predict "col1", "col2", etc.
def constructBaggedTree(df, targetColumn, limit, gainType, T):
    length = df.loc[0].size
    labels = []
    iter = 1
    trees = []



    for i in range(length):
        labels.append("col" + str(i))

    df.columns = labels



    newCol = df[targetColumn]


    data = df.drop(targetColumn, axis=1)
    data[targetColumn] = newCol



    data.columns = labels

    # Here create the bootstrap sample
    newData =  bootstrapSample(data)


    tree = buildTree(newData, newData.keys()[-1], 0, limit, gainType)

    trees.append(tree)

    while iter < T:

        # Next test the tree and and update wweights and run again 
        #run tree through the info again and set predicted value column
        # add predicted value column to data
        # find where prediciotn is not the actual result and sum the wieghts.
        

        #Create another Bootstrap Sample
        newData = bootstrapSample(data)

        tree = buildTree(newData, newData.keys()[-1], 0, limit, gainType)
        trees.append(tree)

        iter = iter + 1

    return trees


# Builds a decision Tree
# df - dataframe to use
# labelName - label to use or column, 'col1', 'col2'
# height - input as 0 to start
# gainType - "ME" - Majority Error, "IG" -information gain, or "GI" - gini index
def buildTreeWithBags(df, labelName, height, limit, gainType, tree=None): 
    
    newNode = get_highest_IG(df, labelName, gainType)

    labelValues = np.unique(df[newNode])

    nodeName = newNode.split('.')[0]





    
    if tree is None:                    
        tree={}
        tree[nodeName] = {}
    
    #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 

    for value in labelValues:
        
        nodetable = get_subtable(df, newNode, value)
        

        clValue, counts = np.unique( nodetable[labelName] , return_counts=True) 

        size = len(clValue)   
        getSame = get_all_same(nodetable, labelName, gainType)

        #Check if there is a repeat on the Node, then we know there is no best answer and we need to take the majority.
        majority = False
        if 'allsame' == getSame and height < limit:
            buildTreeWithBags(nodetable, labelName, height+1, limit, gainType)
        elif 'allsame' == getSame:
            majority = True

            
            
        
                
        # When count length is one we know there is only one label for that subtable in that node 
        if len(counts)==1 or height==limit or majority: 

            if len(counts)==1:
                tree[nodeName][value] = clValue[0]  
            else:
                tree[nodeName][value] =  get_majority(df, labelName)
                                                          
        else:        
            tree[nodeName][value] = buildTreeWithBags(nodetable, labelName, height+1, limit, gainType) #Calling the function recursively 
                   

    return tree



def get_all_same(df, labelName, gainType):
    Entropy_att = []
    IG = []
    labelArray = np.array([labelName])
    keys = df.keys().to_numpy()
    result = np.setdiff1d(keys, labelArray)
    
    for key in df.keys()[:-1]:
        
        
        # Information gain for each attribute and return the key with the highest gain. 
        if gainType == 'ME':
            IG.append(get_total_ME(df, labelName)-find_ME_attribute(df,key,labelName))
        elif gainType == 'GI':
            IG.append(get_total_GI(df, labelName)-find_GI_attribute(df,key,labelName))
        else:
            IG.append(get_total_entropy(df, labelName)-find_entropy_attribute(df,key,labelName))




    baseValue = IG[0]
    allsame = True
    for value in IG:
        if baseValue != value:
            allsame = False

    if allsame:
        return 'allsame'
    return df.keys()[:-1][np.argmax(IG)]








# Constructs a tree with Ada Boost
# df - dataframe to use
# targetColumn - column you want to predict "col1", "col2", etc.
def constructRandomForestTree(df, targetColumn, limit, gainType, T):
    length = df.loc[0].size
    labels = []
    iter = 1
    trees = []



    for i in range(length):
        labels.append("col" + str(i))

    df.columns = labels



    newCol = df[targetColumn]


    data = df.drop(targetColumn, axis=1)
    data[targetColumn] = newCol



    data.columns = labels
    
    newData =  bootstrapSample(data)


    tree = buildTreeWithRandomForest(newData, newData.keys()[-1], 0, limit, gainType)

    trees.append(tree)

    while iter < T:

        # Next test the tree and and update wweights and run again 
        #run tree through the info again and set predicted value column
        # add predicted value column to data
        # find where prediciotn is not the actual result and sum the wieghts.
        



        newData = bootstrapSample(data)
        # Now select the random variables to use


        tree = buildTreeWithRandomForest(newData, newData.keys()[-1], 0, limit, gainType)
        trees.append(tree)
        

        iter = iter + 1

    return trees


# Builds a decision Tree
# df - dataframe to use
# labelName - label to use or column, 'col1', 'col2'
# height - input as 0 to start
# gainType - "ME" - Majority Error, "IG" -information gain, or "GI" - gini index
def buildTreeWithRandomForest(df, labelName, height, limit, gainType, tree=None): 
    

    newNode = get_highest_IG_RandomForest(df, labelName, gainType)

    labelValues = np.unique(df[newNode])

    nodeName = newNode.split('.')[0]





    
    if tree is None:                    
        tree={}
        tree[nodeName] = {}
    
    #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 

    for value in labelValues:
        
        nodetable = get_subtable(df, newNode, value)
        

        clValue, counts = np.unique( nodetable[labelName] , return_counts=True) 

        size = len(clValue)   
        getSame = get_all_same(nodetable, labelName, gainType)

        #Check if there is a repeat on the Node, then we know there is no best answer and we need to take the majority.
        majority = False
        if 'allsame' == getSame and height < limit:
            buildTreeWithRandomForest(nodetable, labelName, height+1, limit, gainType)
        elif 'allsame' == getSame:
            majority = True

            
            
        
                
        # When count length is one we know there is only one label for that subtable in that node 
        if len(counts)==1 or height==limit or majority: 

            if len(counts)==1:
                tree[nodeName][value] = clValue[0]  
            else:
                tree[nodeName][value] =  get_majority(df, labelName)
                                                          
        else:        
            tree[nodeName][value] = buildTreeWithRandomForest(nodetable, labelName, height+1, limit, gainType) #Calling the function recursively 
                   

    return tree

# #Use bootstrap where bagging where m' = m
# def subSampleForForest(dataset):
#     # must retrive a list of column names 
#     newDataFrame = pd.DataFrame()
#     selectables = dataset.keys()[:-1]
#     dataLength = len(dataset)
#     numberOfRandomSelections = random.randint(0, dataLength)
    

#     for i in range(len(selectables)):
#         choice = random.choice(selectables)

#         newDataFrame[choice+'.'+str(i)] = dataset[choice]

#     newDataFrame[dataset.keys()[-1]] = dataset[dataset.keys()[-1]]

#     for i in range(numberOfRandomSelections):
#         dropIndex = random.randint(0, dataLength-1)
#         dataset = dataset.drop([dropIndex])
#         dataset = dataset.reset_index(drop=True)
#         dataLength -= 1

        





        

    return newDataFrame



def testTrees(trees, df, testCol):

    iter = 1
    x = []
    y = []

    for tree in trees:

        array = []
    
        for i in range(len(df)):
            line = pd.Series(df.iloc[i])
            insideArray = []
            insideArray.append(find_result(line, tree)) # contruct a prediction column

            array.append(insideArray)

        newCheckArray = pd.DataFrame(np.array(array))

        df['results'] = newCheckArray
        df['realvalue'] = df[testCol]

        err = len(df[testCol][df[testCol] != df['results']])
        y.append(err)
        x.append(iter)
        
        
        iter += 1

    return x, y

def testEnsemble(trees, df, testCol, alpha):

    array = []
    x = []
    y = []

    for w in range(len(trees)):


        for i in range(len(df)):

            line = pd.Series(df.iloc[i])
            insideArray = []

            suggestions = {}
            for j in range(w+1):

                result = find_result(line, trees[j])
                if result in suggestions:
                    suggestions[result] += alpha[j]
                else:
                    suggestions[result] = alpha[j]
        
            max = -1
            winner = ""
            for key in suggestions.keys():
                if suggestions[key] > max:
                    winner = key
                    max = suggestions[key]
            insideArray.append(winner)
        
            array.append(insideArray)


            newCheckArray = pd.DataFrame(np.array(array))

        df['results'] = newCheckArray
        df['realvalue'] = df[testCol]

        err = len(df[testCol][df[testCol] != df['results']])
        
        x.append(w+1)
        y.append(err)

    return x, y

def testGroupDecision(trees, df, testCol):

    array = []
    x = []
    y = []
    for w in range(len(trees)):


        for i in range(len(df)):

            line = pd.Series(df.iloc[i])
            insideArray = []

            suggestions = {}
            for j in range(w+1):

                result = find_result(line, trees[j])
                if result in suggestions:
                    suggestions[result] += 1
                else:
                    suggestions[result] = 1
        
            max = -1
            winner = ""
            for key in suggestions.keys():
                if suggestions[key] > max:
                    winner = key
                    max = suggestions[key]
            insideArray.append(winner)
        
            array.append(insideArray)


            newCheckArray = pd.DataFrame(np.array(array))

        df['results'] = newCheckArray
        df['realvalue'] = df[testCol]

        err = len(df[testCol][df[testCol] != df['results']])
        x.append(w+1)
        y.append(err)
        

    return x, y

def GetSamplesWithoutReplacement(total, data):

    newDataFrame = pd.DataFrame(columns=data.columns)

    for i in range(total):
        choice = random.randint(0, len(data)-1) # Get random index
        newDataFrame.loc[len(newDataFrame.index)] = data.iloc[choice]
        data.drop(index=choice)# remove the row afterwards


    return newDataFrame

# df = pd.read_csv("/Users/hankgansert/Desktop/ML/MachineLearning/EnsembleLearning/bank/train.csv", header=None)
# dfTest = pd.read_csv("/Users/hankgansert/Desktop/ML/MachineLearning/EnsembleLearning/bank/test.csv", header=None)

# #test = pd.read_csv("/Users/hankgansert/Desktop/ML/MachineLearning/EnsembleLearning/bank/test.csv", header=None)

# df = cleanNumbericalValues(df)
# dfTest = cleanNumbericalValues(dfTest)


# #tree = constructTree(df, 'col16', 0, 'IG')
# #2.a 
# #Get all trees and their alpha values
# trees, alphas = constructAdaTree(df, 'col16', 0, 'IG', 500)
# #Results against training 
# x, y = testTrees(trees, df, 'col16')
# plt.plot(x, y)
# # naming the x axis
# plt.xlabel('Iteration')
# # naming the y axis
# plt.ylabel('Error')
 
# # giving a title to my graph
# plt.title('AdaBoost Individual on Training')
# # function to show the plot
# plt.show()

# x, y = testEnsemble(trees, df, 'col16', alphas)
# plt.plot(x, y)
# # naming the x axis
# plt.xlabel('Iteration')
# # naming the y axis
# plt.ylabel('Error')
 
# # giving a title to my graph
# plt.title('AdaBoost Ensemble on Training')
# # function to show the plot
# plt.show()

# #Results against Test
# x, y = testTrees(trees, dfTest, 'col16')
# plt.plot(x, y)
# # naming the x axis
# plt.xlabel('Iteration')
# # naming the y axis
# plt.ylabel('Error')
 
# # giving a title to my graph
# plt.title('AdaBoost Individual on Test')
# # function to show the plot
# plt.show()

# x, y = testEnsemble(trees, dfTest, 'col16', alphas)
# plt.plot(x, y)
# # naming the x axis
# plt.xlabel('Iteration')
# # naming the y axis
# plt.ylabel('Error')
# # giving a title to my graph
# plt.title('AdaBoost Ensemble on Test')
# # function to show the plot
# plt.show()


# #Results against test
# #testTrees(trees, dfTest, 'col16')
# #testEnsemble(trees, dfTest, 'col16', alphas)

# #2.b #BAGGED TREES




# #Results against test
# # testTrees(baggedTrees, dfTest, 'col16')
# # testGroupDecision(baggedTrees,dfTest,'col16')



# #2.c
# # totalBags = []
# # for i in range(100):
# #     newSample = GetSamplesWithoutReplacement(1000, df)
# #     baggedTrees = constructBaggedTree(newSample, 'col16', 0, 'IG', 100)
# #     totalBags.append(baggedTrees)



# #2.d
# randomForest = constructRandomForestTree(df, 'col16', 4, 'IG', 500)

# x, y = testTrees(randomForest, df, 'col16')
# plt.plot(x, y)
# # naming the x axis
# plt.xlabel('Iteration')
# # naming the y axis
# plt.ylabel('Error')
 
# # giving a title to my graph
# plt.title('Random Forest Individual on Training')
# # function to show the plot
# plt.show()

# x, y = testGroupDecision(randomForest, df, 'col16')
# plt.plot(x, y)
# # naming the x axis
# plt.xlabel('Iteration')
# # naming the y axis
# plt.ylabel('Error')
 
# # giving a title to my graph
# plt.title('Random Forest Ensemble on Training')
# # function to show the plot
# plt.show()

# #Results against Test
# x, y = testTrees(randomForest, dfTest, 'col16')
# plt.plot(x, y)
# # naming the x axis
# plt.xlabel('Iteration')
# # naming the y axis
# plt.ylabel('Error')
 
# # giving a title to my graph
# plt.title('Random Forest Individual on Test')
# # function to show the plot
# plt.show()

# x, y = testGroupDecision(randomForest, dfTest, 'col16')
# plt.plot(x, y)
# # naming the x axis
# plt.xlabel('Iteration')
# # naming the y axis
# plt.ylabel('Error')
# # giving a title to my graph
# plt.title('Random Forest Ensemble on Test')
# # function to show the plot
# plt.show()



