# This file uses ML decision tree techiniques
# Author: Hank Gansert
# Date: 09/11/21
import math
import pandas as pd
import numpy as np
from fractions import Fraction



smallNumber = .000001

def get_majority(df, labelName):

    variables = df[labelName].unique()
    greatestAttr = ''
    greatestSize = -1

    for variable in variables:

        if len(df[labelName][df[labelName]==variable]) > greatestSize:
            greatestSize = len(df[labelName][df[labelName]==variable])
            greatestAttr = variable

    return greatestAttr


def get_total_entropy(df, labelName):

    entropy = 0


    labels = df[labelName].unique() 

    for label in labels:
        fraction = df[labelName].value_counts()[label]/len(df[labelName])
        entropy += -fraction*np.log2(fraction)
    return entropy
  
  
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


def get_highest_IG(df, labelName):
    Entropy_att = []
    IG = []
    labelArray = np.array([labelName])
    keys = df.keys().to_numpy()
    result = np.setdiff1d(keys, labelArray)
    
    for key in df.keys()[:-1]:
        
        
        # Information gain for each attribute and return the key with the highest gain. 
        
        IG.append(get_total_entropy(df, labelName)-find_entropy_attribute(df,key,labelName))


    baseValue = IG[0]
    allsame = True
    for value in IG:
        if baseValue != value:
            allsame = False

    if allsame:
        return 'allsame'
    return df.keys()[:-1][np.argmax(IG)]
  
  
def get_subtable(df, node,value):
  return df[df[node] == value].reset_index(drop=True)


def buildTree(df, labelName, height, tree=None): 
    
    newNode = get_highest_IG(df, labelName)

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

        #Check if there is a repeat on the Node, then we know there is no best answer and we need to take the majority.
        majority = False
        if 'allsame' == get_highest_IG(nodetable, labelName):
            majority = True
            
        
                
        # When count length is one we know there is only one label for that subtable in that node 
        if len(counts)==1 or height==6 or majority: 

            if len(counts)==1:
                tree[newNode][value] = clValue[0]  
            else:
                tree[newNode][value] =  get_majority(df, labelName)
                                                          
        else:        
            tree[newNode][value] = buildTree(nodetable, labelName, height+1) #Calling the function recursively 
                   

    return tree






df = pd.read_csv("/Users/hankgansert/Desktop/ML/MachineLearning/DecisionTree/car/train.csv", header=None)

test = pd.read_csv("/Users/hankgansert/Desktop/ML/MachineLearning/DecisionTree/car/train.csv", header=None)



def constructTree(df, targetColumn):
    length = df.loc[0].size
    labels = []


    for i in range(length):
        labels.append("col" + str(i))

    #print(labels)
    df.columns = labels



    newCol = df[targetColumn]

    print(newCol)
    data = df.drop(targetColumn, axis=1)
    data[targetColumn] = newCol
    print(data)


    #print(labels)
    data.columns = labels
    print(data)

    tree = buildTree(data, data.keys()[-1], 0)
    print(tree)
    return tree

def create_test_frame(df, targetColumn):
    length = df.loc[0].size
    labels = []


    for i in range(length):
        labels.append("col" + str(i))

    #print(labels)
    df.columns = labels



    newCol = df[targetColumn]

    print(newCol)
    data = df.drop(targetColumn, axis=1)
    data[targetColumn] = newCol
    print(data)
    


    #print(labels)
    data.columns = labels

    return data

def predict(inst,tree):
    #This function is used to predict for any input variable 
    
    #Recursively we go through the tree that we built earlier

    for nodes in tree.keys():        
        
        value = inst[nodes]
        tree = tree[nodes][value]
        prediction = 0
            
        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break;                            
        
    return prediction

tree = constructTree(df, 'col4')
#Time to test the data 
def test_data(tree, test):

    
    test = create_test_frame(test, 'col4')
    testCol =test[test.keys()[-1]]
    testCol = pd.DataFrame(testCol)
    test = test.drop(test.keys()[-1], axis=1)

    array = []
    for i in range(len(test)):
        line = pd.Series(test.iloc[i])
        insideArray = []
        insideArray.append(predict(line, tree))
        array.append(insideArray)

    newCheckArray = pd.DataFrame(np.array(array))

    testCol['results'] = newCheckArray
    labels = ['training', 'results']
    testCol.columns = labels

    testCol['hits'] = testCol.apply( lambda row: row.training == row.results, axis=1)




    total = len(testCol)
    hits = len(testCol['hits'][testCol['hits']==True])
    return hits/total

print(test_data(tree, test))
