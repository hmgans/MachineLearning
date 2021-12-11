import numpy as np
from math import exp

def sigma(input):
	return 1.0 / (1.0 + np.exp(-input))

def mult(a, b):
    b = [0,0,0]
    for i in range(3):
        b[i] = a[i] * b[i]

dataset = [[0.5, -1, 0.3,1],[-1,-2,-2,-1],[1.5,0.2,-2.5,1]]
rate = [.01,.005,.0025]
w = [0,0,0]
for r in rate:
    for row in dataset:
        w = np.array(w) + r * (1-sigma(np.array(row[-1]) * np.array(row[:-1]) * np.array(w)))*np.array(row[-1])*np.array(row[:-1]) - np.array(w)

print(w)
