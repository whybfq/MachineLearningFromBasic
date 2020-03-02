# coding: utf-8
"""
one_layer_two_points.py (BSD Licensed)
© 2020 (littlesanner@gmail.com)
"""
import numpy as np

c = np.pi / 2
Bits = 10  # hidden_size


def init_network():
    return dict(
        W1=np.array([np.random.randn(Bits)]),  # W1=np.array([[0.1, 0.3, 0.5]]),
        # b1=np.array(np.random.randn(Bits).astype(int) * c)  # b1=np.array([c, c, c])
        b1=np.zeros(Bits)
    )


# activation function
def h1(x):
    return x


def h2(W, a):  # W is the key1 and a is the output form last layer
    return np.linalg.inv(W.dot(1 / W.T)).dot(a)


def forward(network: dict, x: np.array) -> object:
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

    # W1 is the weight of layer1 (key1)
    # b1 is the bias of layer1 and plan to add b1 to key1
    W1, b1 = network['W1'], network['b1']

    W2 = 1 / W1.T  # W2 = network['W2']
    print(
        f'W1:{W1} and its dimension is {W1.ndim}, its shape {W1.shape}\n'
        # f'W2:{W2} and its dimension is {W2.ndim}, its shape {W2.shape}\n'
    )
    # b2 = -c * np.sum(W2)  # plan to add b2 to key2
    b2 = np.zeros(len(x))
    print('b1:{b1}\nb2: {b2}'.format(b1=b1, b2=b2))

    # layer 1
    a1 = np.dot(x, W1) + b1
    z1 = h1(a1)
    print(
        f'a1: {a1} and its dimension is {a1.ndim}, its shape {a1.shape}\n'
        f'Cipher (z1): {z1} and its dimension is {z1.ndim}, its shape {z1.shape}'
    )  # z1 is the cipher

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

x = np.array([23.0])   # eg: np.array([[1, 2]]), np.array([[[1, 2, 3]]])
print('Input Plaintext : {}'.format(x))

y = forward(network, x)
print('Decrypt text : {}'.format(y))


# test for the file
# cipher_text, key1 = open("Cipher_String.txt", 'r'), open("key1_String.txt", 'r')
# for i in cipher_text.readlines():
#     print("Each message: ", i)
# a, k = np.asarray(cipher_text.read()), np.asarray(key1.read())
# print(f"a is {a}\nkey is {k}")
# for i in key1.readlines():
#     print("Each key: ", i)

# cipher_text2, key2 = open("Cipher_Ndarray.txt", 'r'), open("key1_Ndarray.txt", 'r')
# for i in cipher_text2.readlines():
#     print("Each message in numpy array: ", i)
#
# for i in key2.readlines():
#     print("Each key in numpy array: ", i)
