# coding: utf-8
import os
import sys

# sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
# from common.functions import *
# from common.gradient import numerical_gradient

# from common.functions import *
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[ x >= 0 ] = 1
    return grad


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # in case of overflow
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[ 0 ]
    return -np.sum(np.log(y[ np.arange(batch_size), t ] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=[ 'multi_index' ], op_flags=[ 'readwrite' ])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[ idx ]
        x[ idx ] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[ idx ] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[ idx ] = (fxh1 - fxh2) / (2 * h)

        x[ idx ] = tmp_val  # 还原值
        it.iternext()

    return grad


class Adam:
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[ key ] = np.zeros_like(val)
                self.v[ key ] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[ key ] += (1 - self.beta1) * (grads[ key ] - self.m[ key ])
            self.v[ key ] += (1 - self.beta2) * (grads[ key ] ** 2 - self.v[ key ])

            params[ key ] -= lr_t * self.m[ key ] / (np.sqrt(self.v[ key ]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=1):
        # 初始化权重
        self.params = dict(W1=weight_init_std * np.random.randn(input_size, hidden_size), b1=np.zeros(hidden_size),
                           W2=weight_init_std * np.random.randn(hidden_size, output_size), b2=np.zeros(output_size))

    def predict(self, x):
        W1 = self.params['W1']
        W2 = W1.T
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = relu(a1)

        a2 = np.dot(z1, W2) + b2
        # def h2(a, W1, W2):  # W is the KeyA and a is the output form last layer
        #     return a.dot(np.linalg.inv(W1.dot(W2))) # ! use 'a' to multiply the inverse matrix of the identity matrix
        y = a2.dot(np.linalg.inv(W1.dot(W2)))

        
        return y
        
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return mean_squared_error(y, t)
    
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


if __name__ == '__main__':
    # test simpleNet()
    # net1 = simpleNet()
    # print(net1.W)
    # x = np.array([ 0.6, 0.9 ])
    # p = net1.predict(x)
    # print(p)

    # test TwoLayerNet
    net = TwoLayerNet(input_size=4, hidden_size=5, output_size=4)
    print(f"W1 is {net.params['W1']}")
    x = np.array([[72., 101.,  2.,   5] ])
    t = x  # the tag is the data itself
    print(f'input is {x}')
    y = net.predict(x)
    print(y)
