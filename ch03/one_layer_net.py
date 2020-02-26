# coding: utf-8
"""
one_layer_two_points.py (BSD Licensed)
© 2020 (littlesanner@gmail.com)
"""
import numpy as np

c = np.pi / 2
Bits = 100000


class OneLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size), 'b1': np.zeros(hidden_size),
                       'W2': weight_init_std * np.random.randn(hidden_size, output_size), 'b2': np.zeros(output_size)}

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


def init_network():
    network = dict(W1=np.array([[0.1, 0.3, 0.5]]), b1=np.array([c, c, c]))
    return network

# activation function
def h1(x):
    return x


def h2(W, a):  # W is the key1 and a is the output form last layer
    return np.linalg.inv(W.dot(1 / W.T)).dot(a)


def forward(network, x):
    """
    forward(network, x)

        The main process of the neural network to forward the process of cipher and decrypt

        Parameters
        ----------
        network : array_like
            An class to initialize the W1 and b1 of the neural network
        x : ndarray
            The plain text that input to be decrypted

        Returns
        -------
        out : Decrypt text

        Notes
        -----

        Examples
        --------

    """

    # W1 = np.array([[2, 5, 99]])  # W1 is the weight of layer1 (key1)
    W1 = np.array([np.random.randn(Bits)])

    W2 = 1 / W1.T  # W2 = network['W2']
    print('W1:{W1}, W2: {W2}'.format(W1=W1, W2=W2))
    # b1 = network['b1']  # b1 is the bias of layer1 and it will be used to make the key1 more complicated
    b1 = np.array([np.random.randn(Bits).astype(int) * c])  # plan to add b1 to key1
    b2 = -c * np.sum(W2)  # plan to add b2 to key2
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
