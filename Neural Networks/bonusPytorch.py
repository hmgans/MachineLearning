import tensorflow as tf
import numpy as np


# ReLU
dense1 = tf.keras.layers.Dense(3, activation=tf.nn.relu)
Input1 = tf.keras.Input(shape=(16,))
model2 = tf.Model(dense1,Input1)
#Xavier Initializer
initializer1 = tf.contrib.layers.xavier_initializer()
W1 = tf.get_variable("W", shape=[784, 256],
           initializer=tf.contrib.layers.xavier_initializer())
#Tanh
dense2 = tf.keras.layers.Dense(3, activation=tf.nn.tanh)
Input2 = tf.keras.Input(shape=(16,))

model2 = tf.Model(dense2,Input2)
# HE initializer
W2 = tf.get_variable('W1', shape=[784, 256],
       initializer=tf.contrib.layers.variance_scaling_initializer())


