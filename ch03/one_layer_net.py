# coding: utf-8
import numpy as np

c = np.pi / 2


def init_network():
    network = dict(W1=np.array([[0.1, 0.3, 0.5]]), b1=np.array([c, c, c]))

    return network


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


def h1(x):
    return x


def h2(W):
    return np.linalg.inv(W.dot(1 / W.T))


def forward(network, x):
    # the weight of layer1, layer2 and output layer
    # W1 = np.array([[0.1, 0.3, 0.5]])
    W1 = np.array([np.random.randn(10000)])

    W2 = 1 / W1.T  # W2 = network['W2']
    print('W1:{W1}, W2: {W2}'.format(W1=W1, W2=W2))
    # the bias of layer1, layer2 and output layer
    b1 = network['b1']
    b2 = -c * np.sum(W2)
    print('b1:{b1}\nb2: {b2}'.format(b1=b1, b2=b2))

    # layer 1
    a1 = np.dot(x, W1)
    z1 = h1(a1)
    print('a1: {a1}\nz1: {z1}'.format(a1=a1, z1=z1))

    # output layer
    W2 = 1 / W1.T
    a2 = np.dot(z1, W2)   # w2 should be 1/(w1.T)  # a2 should be 3
    z2 = h2(W1)
    print('a2: {a2} \n'
          'z2 (before the last output): {z2}'.format(a2=a2, z2=np.linalg.inv(W1.dot(W2))))

    # y = z2.dot(a2)
    y = np.linalg.inv(W1.dot(W2)).dot(a2)
    return y


network = init_network()
x = np.array([97.0])
y = forward(network, x)
print(y)
