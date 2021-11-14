# This file uses ML SVM
# Author: Hank Gansert
# Date: 09/11/21
import math
from os import error
import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame


# This is the regular perceptron 
# df - cleaned dataframme
# epoch - number of epochs
# r - step size
def SVM_SG(df, epoch, r):
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

            # Small Change Here for the SVM figure out what N is and what C should be and [w;0]

            if result <= 1:
                w = w -  r * y * x + C 
 
    return w