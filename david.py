"""
This code was written by David Fridlander
Sep 18th, 2020
u1136450
"""

from __future__ import division
from itertools import repeat
import os
import sys
import random
import math
import re

higest_d = 0

class Tree:
    def __init__(self, data):
        self.data = data
        self.child = []
        self.parent = None
    def add_child(self, child):
        child.parent = self
        self.child.append(child)

    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent

        return level

    def print_tree(self):
        #space = ' ' + self.get_level()
        print( self.data)
        if self.child:
            for x in self.child:
                x.print_tree()

#
def ID3(S, Attributes, depth, max_depth):
    global higest_d
    if depth > higest_d:
        higest_d = depth
    tot_ent = totalentro(S)
    #print(tot_ent)
    if tot_ent == 0.0:
        current_label = S[0][0: 2]
        root = Tree(current_label)
        return root

    else:
        entropys = entropy(S, Attributes, tot_ent)
        infogain = infogain_func(S, entropys, tot_ent)
        biggest = biggest_gain(infogain) #attri 
        #take the highest info gain, make that a root
        root = Tree(biggest) #40:1 is the higest root
        for i in range(2):
            Sv = attrlist(S, biggest, i)
            if len(Sv) == 0 or len(Attributes) == 1 or depth == max_depth:
                current_label = S[0][0: 2]
                subset = Tree(current_label) #got the answer
                root.add_child(subset)
            else:
                lst = []
                for x in Attributes:
                    lst.append(x)
                lst.remove(int(biggest))
                subset = ID3(Sv, lst, depth+1, max_depth)
                root.add_child(subset)
        return root

#   
def attrlist(S, big, label):
    lst = []
    if label is 1: #hit
        for line in S:
            if ""+str(big)+":1" in line:
                lst.append(line)
                
    elif label is 0: #miss
        for line in S:
            if not ""+str(big)+":1" in line:
                lst.append(line)

    return lst

#Returns the biggest gain attribute as a string
def biggest_gain(gainlist):
    biggest = max(gainlist, key=gainlist.get)
    return str(biggest)


#returns the total entropy using the current dataset
def totalentro(S):
    pos = 0
    neg = 0
    for line in S:
        label = line[0: 2]
        if label == "+1":
            pos += 1
        elif label == "-1":
            neg += 1

    total = neg + pos
    if neg == 0 or pos == 0:
        totalentropy = 0

    else:
        totalentropy = -(pos/total)*math.log(pos/total, 2.0)-(neg/total)*math.log(neg/total, 2.0)    

    return totalentropy

#returns a dictionary with the the attribute as a key and information gain as a value
def infogain_func(S ,entlist, totalentropy):
    infogain = {}
    for key, value in entlist.items():
        infogain[key] = totalentropy-value
    return infogain

#returns a dictionary with the the attribute as a key and entropy as a value
def entropy(S, Attributes, totalentropy):
    entropylist = {}
    for item in Attributes: #loop attributes times
        key = str(item)
        withattri = [] #list that has the attribute
        withoutattri = [] #list the does NOT have the attribute
        for line in S:
            if ""+key+":1" in line:
                withattri.append(line) 
            else:
                withoutattri.append(line)
        
        total = len(withattri) + len(withoutattri)
        yes = 0
        no = 0
        poslabel = "+1"
        for i in withattri: #part 1 - exists in the line
            if poslabel in i:
                yes += 1
            else:
                no += 1
            
        if yes == 0 or no == 0:
            x = 0
        else:
            x = -(yes/(len(withattri)))*math.log(yes/(len(withattri)), 2.0)-(no/(len(withattri)))*math.log(no/(len(withattri)), 2.0)
           
        yes = 0
        no = 0
        for j in withoutattri: #part 2 - does not exist in the line
            if poslabel in j:
                yes += 1
            else:
                no += 1

        if yes == 0 or no == 0:
            y = 0
        else:
            y = -(yes/len(withoutattri))*math.log(yes/(len(withoutattri)), 2.0)-(no/(len(withoutattri)))*math.log(no/(len(withoutattri)), 2.0)

            #now we have the lists to fine the entropy for each attri
        entropy = x*(len(withattri)/total) + y*(len(withoutattri)/total)
        #print(entropy)
        entropylist[key] = entropy #list of entropies
    return entropylist

#function to find the accuracy of the training
def accuracy(t, test):
    hit = 0
    miss = 0
    for line in test:
        if check_hit(t, line):
            hit += 1
        else:
            miss +=1
     
    total = miss + hit
    return float(hit/total)

#helper for accuracy function, finds if there the tree got the right data
def check_hit(t, line):
    pos_label = "+1"
    neg_label = "-1"
    #print t.data
    if t.data == pos_label or t.data == neg_label: #base case
        #print "im here"
        if t.data == line[0: 2]: #we got a hit
            return True
        else: #we missed
            return False
    else: #need to check if the data is in the line
        attri = " "+str(t.data)+":1 "
        #print attri
        if len(t.child) > 1:
            if attri in line:
                return check_hit(t.child[1], line)
            else:
                return check_hit(t.child[0], line)

        elif len(t.child) == 1:
            return check_hit(t.child[0], line)

        else:
            if t.data == line[0: 2]:
                return True
            else:
                return False
                
def cross_validation(f1, f2, f3, f4, f5, Attributes, higest, t):
    depth = 1
    acc = 0
    biggest = 0
    biggest_avg = 0
    while depth <= higest:
        acc = 0
        lst = []
        lst.extend(f1+f2+f3+f4)
        tree = ID3(lst, Attributes, 0, depth)
        acc += accuracy(tree, f5)
        
        lst = []
        lst.extend(f2+f3+f4+f5)
        tree = ID3(lst, Attributes, 0, depth)
        acc += accuracy(tree, f1)
        
        lst = []
        lst.extend(f3+f4+f5+f1)
        tree = ID3(lst, Attributes, 0, depth)
        acc += accuracy(tree, f2)

        lst = []
        lst.extend(f4+f5+f1+f2)
        tree = ID3(lst, Attributes, 0, depth)
        acc += accuracy(tree, f3)
        
        lst = []
        lst.extend(f5+f1+f2+f3)
        tree = ID3(lst, Attributes, 0, depth)
        acc += accuracy(tree, f4)

        if t is True:
            print("  Depth: " + str(depth) + " Accuracy: " +  str(acc/5))
        
        if biggest_avg < acc/5:
            biggest = depth
            biggest_avg = acc/5
        depth += 1
    
    if t is True:
        print ("the biggest average for cross validation was: " + str(biggest_avg))
        print ("(g) Best depth: " + str(biggest))
    return biggest
   
    
    
def main():
    train = open('a1a.train', 'r')
    buf = train.readlines()
    Attributes = range(1,124)
    pos = 0
    neg = 0
    for line in buf:
        label = line[0: 2]
        if label == "+1":
            pos += 1
        elif label == "-1":
            neg += 1
    #(a)
    if pos > neg:
        print ("(a) Most common label: "+"+1")
    else:
        print ("(a) Most common label: "+"-1")

    total = neg + pos
    if neg == 0 or pos == 0:
        totalentropy = 0

    else:
        totalentropy = -(pos/total)*math.log(pos/total, 2.0)-(neg/total)*math.log(neg/total, 2.0)
    #(b)
    print ("(b) Entropy: "+ str(totalentropy))
    
    #(c)
    c = open('a1a.train', 'r')
    c1 = c.readlines()
    ent = entropy(c1, Attributes, totalentropy)
    gain = infogain_func(train, ent, totalentropy) #dic of gain
    big_gain = gain[biggest_gain(gain)]
    print ("(c) Best feature: " + str(biggest_gain(gain)+ ":1" + " the gain was: "+ str(big_gain)))
    
    #(d)
    f2 = open('a1a.train', 'r')
    buf = f2.readlines()
    tree = ID3(buf, Attributes, 0, -1)
    acc = accuracy(tree, buf)
    print ("(d) Accuracy on the training set: " + str(acc)) 
    
    #(e)
    test = open('a1a.test', 'r')    
    test_read = test.readlines()
    acc = accuracy(tree, test_read)
    print ("(e) Accuracy on the test set: " + str(acc))
        
    #(f)
    fold1 = open('fold1' , 'r')
    fold2 = open('fold2' , 'r')
    fold3 = open('fold3' , 'r')
    fold4 = open('fold4' , 'r')
    fold5 = open('fold5' , 'r')    
    buf1 = fold1.readlines()
    buf2 = fold2.readlines()
    buf3 = fold3.readlines()
    buf4 = fold4.readlines()
    buf5 = fold5.readlines()

    #lst = []
    print("(f) Average accuracies from cross-validation for each depth:  ")
    x =  cross_validation(buf1, buf2, buf3, buf4, buf5, Attributes, 40, True)
    tree = ID3(buf, Attributes, 0, x)
    test = open('a1a.test', 'r')
    test_read = test.readlines()
    acc = accuracy(tree, test_read)
    print("(h) Accuracy on the test set using the best depth: " + str(acc))

    x =  cross_validation(buf1, buf2, buf3, buf4, buf5, Attributes, 40, False)
    tree = ID3(buf, Attributes, 0, x)
    test = open('a1a.train', 'r')
    test_read = test.readlines()
    acc = accuracy(tree, test_read)
    print("Accuracy on the train set using the best depth: " + str(acc))
    
    #(h)
    
if __name__=="__main__": 
    main()
