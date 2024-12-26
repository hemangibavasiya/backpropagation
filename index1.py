import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and preprocess the data
data = pd.read_csv('fertility.csv').dropna()

# Split into input features (X) and target labels (Y)
Y_label = data['output']
X_feat = data.loc[:, 'season':'hrs_spents_sitting']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_feat, Y_label, test_size=0.3, random_state=42)

# Convert to numpy arrays
x_train_np, y_train_np = x_train.to_numpy(), y_train.to_numpy()
x_test_np, y_test_np = x_test.to_numpy(), y_test.to_numpy()

# Neural Network Configuration
n_features = x_train_np.shape[1]
n_hidden = 8
n_outputs = 1
learning_rate = 0.01
epochs = 100

# Helper functions
def initialize_network(n_features, n_hidden, n_outputs):
    """Initialize the neural network with random weights."""
    network = [
        [{'weights': np.random.rand(n_features + 1)} for _ in range(n_hidden)],  # Hidden layer 1
        [{'weights': np.random.rand(n_hidden + 1)} for _ in range(n_hidden)],    # Hidden layer 2
        [{'weights': np.random.rand(n_hidden + 1)} for _ in range(n_outputs)]    # Output layer
    ]
    return network

def activate(weights, inputs):
    """Compute neuron activation."""
    return np.dot(weights[:-1], inputs) + weights[-1]

def sigmoid(x):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(output):
    """Derivative of the sigmoid function."""
    return output * (1.0 - output)

def forward_propagate(network, row):
    """Forward propagate inputs through the network."""
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def backward_propagate_error(network, expected):
    """Backward propagate the error and store deltas in neurons."""
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network) - 1:  # Not the output layer
            for j in range(len(layer)):
                error = sum([neuron['weights'][j] * neuron['delta'] for neuron in network[i + 1]])
                errors.append(error)
        else:  # Output layer
            for j, neuron in enumerate(layer):
                errors.append(expected[j] - neuron['output'])
        for j, neuron in enumerate(layer):
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])

def update_weights(network, row, l_rate):
    """Update neuron weights with error delta."""
    for i in range(len(network)):
        inputs = row if i == 0 else [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

def train_network(network, train, labels, l_rate, epochs):
    """Train the neural network for a fixed number of epochs."""
    for epoch in range(epochs):
        total_error = 0
        for row, label in zip(train, labels):
            outputs = forward_propagate(network, row)
            expected = [label]
            total_error += sum((expected[j] - outputs[j]) ** 2 for j in range(len(expected)))
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print(f"Epoch {epoch + 1}/{epochs}, Error: {total_error:.5f}")

# Initialize and train the network
network = initialize_network(n_features, n_hidden, n_outputs)
train_network(network, x_train_np, y_train_np, learning_rate, epochs)

# Evaluate the network
def predict(network, row):
    """Make a prediction for a single row."""
    outputs = forward_propagate(network, row)
    return outputs[0]

predictions = [predict(network, row) for row in x_test_np]
print("Predictions:", predictions)
