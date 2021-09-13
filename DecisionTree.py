# This file uses ML decision tree techiniques
# Author: Hank Gansert
# Date: 09/11/21
import math



def entropy(PropPos):
    PropNeg = 1 - PropPos
    return -PropPos * math.log2(PropPos) - PropNeg * math.log2(PropNeg)





print(entropy(5/14))