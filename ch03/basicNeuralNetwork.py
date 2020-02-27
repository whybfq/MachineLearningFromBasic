# coding: utf-8
"""
basicNeuralNetwork.py (BSD Licensed) : only use one layer
© 2020 (littlesanner@gmail.com)
version: 0.0.1
"""

import time
import numpy as np
import pyperclip

c = np.pi / 2
Bits = 10  # hidden_size


# decorator to calculate duration taken by any function.
def calculate_time(func):
    # added arguments inside the inner1,
    # if function takes any arguments,
    # can be added like this.
    def inner1(*args, **kwargs):
        # storing time before function execution
        begin = time.time()

        func(*args, **kwargs)

        # storing time after function execution
        end = time.time()
        print(f"{func.__name__} function total time taken in : {end - begin} seconds")

    return inner1


# every possible symbol that can be encrypted
# LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # if only consider 26 English alphabets
LETTERS = """ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"""
# note the space at the front


class OneLayerNet:

    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std=0.01):
        # initialize the weight and bias
        self.params = dict(
            W1=weight_init_std * np.random.randn(input_size, hidden_size),
            b1=np.zeros(hidden_size),
            b2=np.zeros(output_size)  # the size if the same as input_size
        )

    def h1(self, x):  # activation function 1
        return x

    def h2(self, a, W):  # W is the key1 and a is the output form last layer
        return a.dot(np.linalg.inv(W.dot(1 / W.T)))  # have to use a to multiply the inverse matrix of the identity matrix

    def predict(self, x):
        W1, b1, b2 = self.params['W1'], self.params['b1'], self.params['b2']
        W2 = 1 / W1.T  # W2 is from W1
        print(
            "The dimension of W1 is {d1}\n"
            "And W1(key1) is: {W1}\n"
            "The dimension of W2 is {d2}\n"
            "And W2(key2) is: {W2}".format(d1=W1.shape, W1=W1, d2=W2.shape, W2=W2)
        )
        # print(f'b1:{b1}\nb2: {b2}')

        a1 = np.dot(x, W1) + b1
        z1 = self.h1(a1)  # z1 is the cipher
        print(
            # "The dimension of a1 is: {}\n"
            # "And a1 is: {a1}"
            "Cipher text(z1) : \n{z1}".format(a1.shape, a1=a1, z1=z1)
        )

        a2 = np.dot(z1, W2) + b2
        z2 = self.h2(a2, W1)  # z2 = np.linalg.inv(W1.dot(W2)).dot(a2) is the Decrypted text

        # print(
        #     "The dimension of a2 is: {d1}\n"
        #     "And a2 is: {a2}\n"
        #     "The dimension of z2 is {d2}\n"
        #     "And z2(Decrypt text) is: {z2}\n".format(d1=a2.shape, a2=a2, d2=z2.shape, z2=z2)
        # )

        return z2


@calculate_time
def main():
    """the main process"""
    translated = ""  # stores the encrypted/decrypted form of the message
    # tells the program to encrypt or decrypt
    mode = "encrypt"  # set to 'encrypt' or 'decrypt' mode

    # message = np.array([*string.ascii_lowercase])[3]  #  array(['d'], dtype='<U1')
    Inputs = "a"  # ord('a')->97,  chr(97)->a
    # Inputs = Inputs.upper()  # capitalize the string in message if only consider 26 big alphabets
    print(f"Input Plaintext {Inputs}\n")

    # run the encryption/decryption code on each symbol in the message string
    for symbol in Inputs:
        if symbol in LETTERS:
            message = np.array([ord(symbol)])  # eg: np.array([[1, 2]]), np.array([[[1, 2, 3]]]),
            # print(f"Input Plaintext {Inputs}\nAnd the dimension is: {message.shape}")

            if mode == "encrypt":
                # initialize the neural network
                input_size, hidden_size, output_size = message.ndim, Bits, message.ndim
                test: OneLayerNet = OneLayerNet(input_size, hidden_size, output_size)

                Decrypted_text = test.predict(message)
                Decrypted_text = chr(Decrypted_text.__int__())
                # print(f'Decrypt text : {Decrypted_text}')

            elif mode == 'decrypt':
                pass

            else:
                print('mode can only be encrypt or decrypt!')

            # add encrypted/decrypted number's symbol at the end of translated
            translated += Decrypted_text

        else:
            # just add the symbol without encrypting/decrypting
            translated += symbol

    # print the encrypted/decrypted string to the screen
    print(f"The cracking code is {translated} ")

    # copy the encrypted/decrypted string to the clipboard
    pyperclip.copy(translated)


if __name__ == '__main__':
    main()
