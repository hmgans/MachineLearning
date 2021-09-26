# This file uses ML decision tree techiniques
# Author: Hank Gansert
# Date: 09/11/21
import math
import pandas as pd
import numpy as np



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



        

    






    return 0 

df = pd.read_csv("car/train.csv", header=None)

test = pd.read_csv("car/test.csv", header=None)

# df = cleanNumbericalValues(df)
# test = cleanNumbericalValues(test)

length = df.loc[0].size
labels = []

for i in range(length):
    labels.append("col" + str(i))



# print("Results for GI with unknown")
# tree = constructTree(df, labels[-1], 16, 'GI')
# print(test_data(tree, test, labels[-1]))

# print("Results for GI with unknown")
# tree = constructTree(df, labels[-1], 16, 'ME')
# print(test_data(tree, test, labels[-1]))

# print("Results for GI with unknown")
# tree = constructTree(df, labels[-1], 16, 'IG')
# print(test_data(tree, test, labels[-1]))

# df = cleanUnknownValues(df)
# test = cleanUnknownValues(test)

# print("Results for GI replacing unknown")
# tree = constructTree(df, labels[-1], 6, 'GI')
# print(test_data(tree, test, labels[-1]))

# print("Results for GI replacing unknown")
# tree = constructTree(df, labels[-1], 6, 'ME')
# print(test_data(tree, test, labels[-1]))

# print("Results for GI replacing unknown")
# tree = constructTree(df, labels[-1], 6, 'IG')
# print(test_data(tree, test, labels[-1]))





for i in range(6):
    print("Tree Depth")
    print(i)
    print("Results for GI")
    for label in labels:
        tree = constructTree(df, label, i, 'GI')
        print(label+ ": results")
        print(test_data(tree, test, label))

    print("Results for ME")
    for label in labels:
        tree = constructTree(df, label, i, 'ME')
        print(label+ ": results")
        print(test_data(tree, test, label))

    print("Results for Info Gain")
    for label in labels:
        tree = constructTree(df, label, i, 'IG')
        print(label+ ": results")
        print(test_data(tree, test, label))


