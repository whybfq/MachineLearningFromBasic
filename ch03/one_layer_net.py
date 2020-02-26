# coding: utf-8
"""
one_layer_two_points.py (BSD Licensed)
© 2020 (littlesanner@gmail.com)
"""
import numpy as np

c = np.pi / 2


def init_network():
    network = dict(W1=np.array([[0.1, 0.3, 0.5]]), b1=np.array([c, c, c]))
    return network


def h1(x):
    return x


def h2(W, a):
    return np.linalg.inv(W.dot(1 / W.T)).dot(a)


def forward(network, x):
    # the weight of layer1, layer2 and output layer
    W1 = np.array([[2, 5, 99]])  # w1 is the key1
    # W1 = np.array([np.random.randn(10000)])

    W2 = 1 / W1.T  # W2 = network['W2']
    print('W1:{W1}, W2: {W2}'.format(W1=W1, W2=W2))
    # the bias of layer1, layer2 and output layer
    b1 = network['b1']  # b1 is the bias and it will be used to make the key1 more complicated
    b2 = -c * np.sum(W2)  #
    print('b1:{b1}\nb2: {b2}'.format(b1=b1, b2=b2))

    # layer 1
    a1 = np.dot(x, W1) + b1
    z1 = h1(a1)
    print('a1: {a1}\nCipher (z1): {z1}'.format(a1=a1, z1=z1))  # z1 is the cipher

    # output layer
    W2 = 1 / W1.T
    a2 = np.dot(z1, W2) + b2   # w2 should be 1/(w1.T)  # a2 should be 3
    z2 = h2(W1, a2)  # need to give the key(W1) and a2 to activation function
    print('a2: {a2} \n'
          'z2 (Decrypt text) : {z2}'.format(a2=a2, z2=z2)
          )

    # y = np.linalg.inv(W1.dot(W2)).dot(a2) is the Decrypt text
    return z2


network = init_network()

x = np.array([23.0])
print('Input Plaintext : {}'.format(x))

y = forward(network, x)
# print('Decrypt text : {}'.format(y))
