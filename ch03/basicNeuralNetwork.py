import numpy as np
c = np.pi / 2
Bits = 100000  # hidden_size


def h1(x):  # activation function 1
    return x


def h2(W, a):  # W is the key1 and a is the output form last layer
    return np.linalg.inv(W.dot(1 / W.T)).dot(a)


class OneLayerNet:

    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std=0.01):
        # initialize the weight and bias
        self.params = dict(
            W1=weight_init_std * np.random.randn(input_size, hidden_size),
            b1=np.zeros(hidden_size)
        )

    def predict(self, x):
        W1, b1 = self.params['W1'], self.params['b1']  # W1 is the key1, consider to add b1 inside
        W2 = 1 / W1.T  # W2 is from W1
        b2 = np.zeros(len(x))  # the size if the same as input_size

        a1 = np.dot(x, W1) + b1
        z1 = h1(a1)

        a2 = np.dot(z1, W2) + b2
        y = h2(W1, a2)

        print(y)
        return y


# initialize the neural network
input_size, hidden_size, output_size = 1, Bits, 1
test: OneLayerNet = OneLayerNet(input_size, hidden_size, output_size)

plaintext = np.array([97])
print('Input Plaintext : {}'.format(plaintext))

Decrypted_text = test.predict(plaintext)
print('Decrypt text : {}'.format(Decrypted_text))

