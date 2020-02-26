import numpy as np

# c = np.pi / 2
Bits = 100000  # hidden_size


def h1(x):  # activation function 1
    return x


def h2(W, a):  # W is the key1 and a is the output form last layer
    return a.dot(np.linalg.inv(W.dot(1 / W.T)))   # have to use a to multiply the inverse matrix of the identity matrix


class OneLayerNet:

    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std=0.01):
        # initialize the weight and bias
        self.params = dict(
            W1=weight_init_std * np.random.randn(input_size, hidden_size),
            b1=np.zeros(hidden_size),
            b2=np.zeros(output_size)  # the size if the same as input_size
        )

    def predict(self, x):
        W1, b1, b2 = self.params['W1'], self.params['b1'], self.params['b2']
        W2 = 1 / W1.T  # W2 is from W1
        print(
            "The dimension of W1 is {0}\n"
            "And W1 is: {1}\n"
            "The dimension of W2 is {2}\n"
            "And W2 is: {3}"
            .format(W1.shape, W1, W2.shape, W2)
        )
        # print('b1:{b1}\nb2: {b2}'.format(b1=b1, b2=b2))

        a1 = np.dot(x, W1) + b1
        z1 = h1(a1)
        print(
            "The dimension of a1 is: {}\n"
            "And a1 is: {a1}"
            "Cipher(z1): {z1}".format(a1.shape, a1=a1, z1=z1)
        )  # z1 is the cipher

        a2 = np.dot(z1, W2) + b2
        z2 = h2(W1, a2)  # y = np.linalg.inv(W1.dot(W2)).dot(a2) is the Decrypt text

        print(
            "The dimension of a2 is: {d1}\n"
            "And a2 is: {a2}\n"
            "The dimension of z2 is {d2}\n"
            "And z2(Decrypt text) is: {z2}\n".format(d1=a2.shape, a2=a2, d2=z2.shape, z2=z2)
        )  # z2 is the decrypted text

        return z2


plaintext = np.array([[7, 9]])
print('Input Plaintext {} and the dimension is: {}'.format(plaintext, plaintext.shape))

# initialize the neural network
input_size, hidden_size, output_size = plaintext.ndim, Bits, plaintext.ndim
test: OneLayerNet = OneLayerNet(input_size, hidden_size, output_size)

Decrypted_text = test.predict(plaintext)
print('Decrypt text : {}'.format(Decrypted_text))
