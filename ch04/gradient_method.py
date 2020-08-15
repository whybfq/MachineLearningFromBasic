# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_1d(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[ idx ]
        x[ idx ] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[ idx ] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[ idx ] = (fxh1 - fxh2) / (2 * h)

        x[ idx ] = tmp_val  # 还原值

    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[ idx ] = _numerical_gradient_1d(f, x)

        return grad


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


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = [ ]

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[ 0 ] ** 2 + x[ 1 ] ** 2 + x[ 2 ] ** 2


init_x = np.array([ -3.0, 4.0, 5.0 ])

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

# 2d dimension
# plt.plot([ -5, 5 ], [ 0, 0 ], '--b')
# plt.plot([ 0, 0 ], [ -5, 5 ], '--b')
# plt.plot(x_history[ :, 0 ], x_history[ :, 1 ], 'o')
#
# plt.xlim(-3.5, 3.5)
# plt.ylim(-4.5, 4.5)
# plt.xlabel("X0")
# plt.ylabel("X1")
# plt.show()


# 3d dimension
# gradient figure
data = x_history
x, y, z = data[ :, 0 ], data[ :, 1 ], data[ :, 2 ]
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(x[ :20 ], y[ :20 ], z[ :20 ], c='y')  # 绘制数据点
# ax.scatter(x[10:20], y[10:20], z[10:20], c='r')
ax.plot3D(x, y, z, 'gray')  # 绘制空间曲线

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()
