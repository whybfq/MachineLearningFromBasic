import numpy as np


def init_network():
    network = {'W1': np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]), 'b1': np.array([0.1, 0.2, 0.3]),
               'W2': np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]), 'b2': np.array([0.1, 0.2]),
               'W3': np.array([[0.1, 0.3], [0.2, 0.4]]), 'b3': np.array([0.1, 0.2])}

    return network


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(network, x):
    # the weight of layer1, layer2 and output layer
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # the bias of layer1, layer2 and output layer
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # layer 1
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    # layer 2
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    # output layer
    a3 = np.dot(z2, W3) + b3
    y = a3
    # test for the ntermediate results
    print('layer1: a1: {a1}, z1: {z1}; \n'
          'layer2: a2: {a2}, z2: {z2}; \n'
          'Output: a3: {a3}, z3: {z3}'.format(a1=a1, a2=a2, a3=a3, z1=z1, z2=z2, z3=y))
    return y


def softmax(a):
    c = np.max(a)
    return np.exp(a - c) / np.sum(np.exp(a - c))


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
