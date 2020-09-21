# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:44:39 2020

@author: Hemangi Bavasiya
"""

import numpy as np
from random import random
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv ('fertility.csv')
data = data.dropna ()
print (type (data))
# Next we will split the report dataset into input x and output Y
Y_label = data['output']  # we will predict the median house value column
# the remainder of the coumns will be used to predict Y
# Select from the  longitude column to the 'median_income' column
X_feat = data.loc[:, 'season':'hrs_spents_sitting']

x_train, x_test, y_train, y_test = train_test_split (X_feat, Y_label, test_size=0.3)

x_train_np = x_train.to_numpy ()
y_train_np = y_train.to_numpy ()

# convert the testing data
x_test_np = x_test.to_numpy ()
y_test_np = y_test.to_numpy ()

# convert numpy array to list
x_train_np = x_train_np.tolist ()
y_train_np = y_train_np.tolist ()

x_test_np = x_test_np.tolist ()
y_test_np = y_test_np.tolist ()

no_feat = np.size (x_train_np, 1)
epochs = 1
no_inst = np.size (x_train_np, 0)
n_hidden = 8
n_outputs = 1


def activate(weights, inputs):
    activation = weights[-1]
    for i in range (len (weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def get_weights():
    network = list ()
    hidden_layer1 = [{'weights': [random () for i in range (no_feat + 1)]} for i in range (n_hidden)]
    network.append (hidden_layer1)
    hidden_layer2 = [{'weights': [random () for i in range (n_hidden + 1)]} for i in range (n_hidden)]
    network.append (hidden_layer2)
    hidden_layer3 = [{'weights': [random () for i in range (n_hidden + 1)]} for i in range (n_hidden)]
    network.append (hidden_layer3)
    hidden_layer4 = [{'weights': [random () for i in range (n_hidden + 1)]} for i in range (n_hidden)]
    network.append (hidden_layer4)
    output_layer = [{'weights': [random () for i in range (n_hidden + 1)]} for i in range (n_outputs)]
    network.append (output_layer)
    return network


generated_inputs = []

for epoch in range (epochs):

    for row in x_train_np:
        network = get_weights ()
        inputs = row
        for layer in network:
            new_inputs = []
            for neuron in layer:
                H = activate (neuron['weights'], inputs)
                H = 1.0 / (1.0 + np.exp (-H))
                new_inputs.append (H)
        record = {
            "output": new_inputs[0],
            "weights": layer[0]['weights']
        }
        generated_inputs.append (record)



def transfer_derivative(output):
    return output * (1.0 - output)

final_error = []

for epoch in range (epochs):
    for row in y_train_np:
        for i in reversed (range (len (generated_inputs))):
            layer = []
            layer.append(generated_inputs[i])
            errors = list ()
            if i != len (generated_inputs) - 1:
                for j in range (len (layer)):
                    error = 0.0
                    for neuron in generated_inputs[i + 1]:
                        error += (generated_inputs[i + 1]['weights'][j] * generated_inputs[i + 1]['delta'])
                    errors.append (error)
            else:
                for j in range (len (layer)):
                    neuron = layer[j]
                    errors.append (y_train_np[j] - neuron['output'])
            for j in range (len (layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * transfer_derivative (neuron['output'])
            print(errors)
            final_error.append(errors)


def update_weights(network, row, l_rate):
    for i in range (len (network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range (len (inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


print(generated_inputs)