import math
import pandas as pd
import numpy as np
from random import randrange
import random
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import Bagging

df = pd.read_csv("/Users/hankgansert/Desktop/ML/MachineLearning/EnsembleLearning/default_of_credit_card_clients.csv", header=None)


df = Bagging.cleanNumbericalValues(df)

Train = Bagging.GetSamplesWithoutReplacement(24000, df)
Test = df
baggedTrees = Bagging.constructBaggedTree(df, 'col16', 3, 'IG', 500)


