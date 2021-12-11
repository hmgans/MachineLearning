# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import pandas as pd
import numpy as np
 
# Class to represent nodes in the layer
class node:

    def __init__(self, weights):

        self.weights = weights
        self.weights_deriv = 0
        self.value = 1

    


def clean(df):
    length = df.loc[0].size
    df[4] = df[4].apply(lambda x: -1 if x==0 else 1 )
    df.insert(loc=4,column='4',value=1)	
    return df



 

def accuracy(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
 

 

def neuron_input(weights, inputs):

	input = weights[-1]
	for i in range(len(weights)-1):
		input += weights[i] * inputs[i]
	return input
 

def sigma(input):
	return 1.0 / (1.0 + exp(-input))
 


 

def sigma_deriv(z):
	return z * (1.0 - z)
 

def backward(NN, expected):

	# y - y*
	layer = NN[len(NN)-1]
	errors = list()
	for i in range(len(layer)):
		node = layer[i]
		errors.append(node.value - expected)

	for i in range(len(layer)):

		# Update the new weights for the layer 
		node = layer[i]
		# weight error * z(1-z)
		node.weights_deriv = errors[i] * sigma_deriv(node.value)


	for i in reversed(range(len(NN)-1)):
		layer = NN[i]
		errors = list()

		for j in range(len(layer)):

			error = 0.0

			# Get the error from the layer above
			for node in NN[i + 1]:

				# Get the error from 
				error += (node.weights[j] * node.weights_deriv)
			errors.append(error)
		for j in range(len(layer)):
			# Update the new weights for the layer 
			node = layer[j]
			# weight error * z(1-z)
			node.weights_deriv = errors[j] * sigma_deriv(node.value)
 

def stochastic(NN, row, l_rate):
	for i in range(len(NN)):

		inputs = row[:-1]

		# first row needs to be updated first
		if i != 0:
			inputs = [node.value for node in NN[i - 1]]

		
		for node in NN[i]:
			for j in range(len(inputs)):

				# Mutiply weights witht the new weight derrivitive
				# Stochastic Gradient
				node.weights[j] -= l_rate * node.weights_deriv * inputs[j]

			# Node with no connections
			node.weights[-1] -= l_rate * node.weights_deriv
 

def train_network(NN, train, l_rate, n_epoch):
	for epoch in range(n_epoch):
		shuffled = pd.DataFrame(train)
		shuffled = shuffled.sample(frac=1)
		result = []
		for row in shuffled.values:
			result.append(list(row))



		for row in result:
			forward(NN, row)
			backward(NN, row[-1])

			stochastic(NN, row, l_rate)
 

def neural_net(width):
	
	NN = list()
	layer_1 = [node([np.random.normal() for i in range(5)]) for i in range(width)]
	NN.append(layer_1)

	layer_2 = [node([np.random.normal() for i in range(width + 1)]) for i in range(width)]
	NN.append(layer_2)

	layer_3 = [node([np.random.normal() for i in range(width + 1)])]
	NN.append(layer_3)
	return NN


def neural_net_0(width):
	
	NN = list()
	layer_1 = [node([0 for i in range(5)]) for i in range(width)]
	NN.append(layer_1)

	layer_2 = [node([0 for i in range(width + 1)]) for i in range(width)]
	NN.append(layer_2)

	layer_3 = [node([0 for i in range(width + 1)])]
	NN.append(layer_3)
	return NN

def forward(NN, row):
	inputs = row
	# Go through all the layers in the network
	for layer in NN:
		input_values = []
		# Find output for each node in the layer
		for node in layer:



			value = neuron_input(node.weights, inputs)


			node.value = sigma(value)

			input_values.append(node.value)

		inputs = input_values
	return inputs


def predict(network, row):
	result = forward(network, row)
	if result[0] > .5:
		return 1
	else:
		return -1
 

def PartB():

	widths = [5,10,25,50,100]
	df = pd.read_csv("bank-note/train.csv", header=None)
	df = clean(df)

	t = pd.read_csv("bank-note/test.csv", header=None)
	t = clean(t)

	train = []
	for row in df.values:
		train.append(list(row))

	test = []
	for row in t.values:
		test.append(list(row))

	for width in widths:
		print("Width: "+str(width))

		l_0 = 0.3
		d = 1
		n_epoch = 10
		width = 5
		network = neural_net(width)
		train_network(network, train, l_0, n_epoch)

		predictions = list()
		for row in train:
			prediction = predict(network, row)
			predictions.append(prediction)
		
		actual = [row[-1] for row in train]
		percent_accuracy = accuracy(actual, predictions)
		print('Accuracy for Train: %s' % percent_accuracy)

		predictions = list()
		for row in test:
			prediction = predict(network, row)
			predictions.append(prediction)
		actual = [row[-1] for row in test]
		percent_accuracy = accuracy(actual, predictions)
		print('Accuracy for Test: %s' % percent_accuracy)

def PartC_InitW0():

	widths = [5,10,25,50,100]
	df = pd.read_csv("bank-note/train.csv", header=None)
	df = clean(df)

	t = pd.read_csv("bank-note/test.csv", header=None)
	t = clean(t)

	train = []
	for row in df.values:
		train.append(list(row))

	test = []
	for row in t.values:
		test.append(list(row))

	for width in widths:
		print("Width: "+str(width))

		l_0 = 0.3
		d = 1
		e = 1
		width = 5
		network = neural_net_0(width)
		train_network(network, train, l_0, e)

		predictions = list()
		for row in train:
			prediction = predict(network, row)
			predictions.append(prediction)
		
		actual = [row[-1] for row in train]
		percent_accuracy = accuracy(actual, predictions)
		print('Accuracy for Train: %s' % percent_accuracy)

		predictions = list()
		for row in test:
			prediction = predict(network, row)
			predictions.append(prediction)
		actual = [row[-1] for row in test]
		percent_accuracy = accuracy(actual, predictions)
		print('Accuracy for Test: %s' % percent_accuracy)


PartB()

PartC_InitW0()



