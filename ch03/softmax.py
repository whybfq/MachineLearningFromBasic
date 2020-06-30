# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def soft_max_improved(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a)


# X = np.arange(-5.0, 5.0, 0.1)
# Y = softmax(X)
# plt.plot(X, Y)
# # plt.ylim(-0.1, 1.1)
# plt.show()


# Test:
a = np.array([0.3, 2.9, 4.0])
# a1 = np.array([1010, 1000, 990]) # overflow

exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

print(exp_a / sum_exp_a)
