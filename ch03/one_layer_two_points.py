# coding: utf-8
"""
one_layer_two_points.py (BSD Licensed)
© 2020 (littlesanner@gmail.com)
"""
import numpy as np


def init_network():
    network = {
        'W1': np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]), 'b1': np.array([0.1, 0.2, 0.3]),
        # 'W2': np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]), 'b2': np.array([0.1, 0.2]),
    }

    return network


def h1(x):
    return x


def h2(x, W):
    return x.dot(np.linalg.inv(W.dot(1 / W.T)))


def forward(network, x):
    # the weight of layer1, layer2 and output layer
    W1 = network['W1']

    # the bias of layer1, layer2 and output layer
    # b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # layer 1
    a1 = np.dot(x, W1)
    z1 = h1(a1)

    # layer 2
    W2 = 1 / W1.T
    a2 = np.dot(z1, W2)

    y = h2(a2, W1)  # y = a2.dot(np.linalg.inv(W1.dot(W2)))

    # output layer
    # a3 = np.dot(z2, W3) + b3
    # y = a3

    print('layer1: a1: {a1}, z1: {z1}; \n'
          'layer2: a2: {a2}, z2: {z2}; \n'.format(a1=a1, a2=a2, z1=z1, z2=y))
    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
