# coding: utf-8
"""
one_layer_two_points.py (BSD Licensed)
© 2020 (littlesanner@gmail.com)
"""
import numpy as np
import string
import pyperclip

c = np.pi / 2
Bits = 100000  # hidden_size


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
        # print(
        #     "The dimension of W1 is {0}\n"
        #     "And W1 is: {1}\n"
        #     "The dimension of W2 is {2}\n"
        #     "And W2 is: {3}"
        #     .format(W1.shape, W1, W2.shape, W2)
        # )
        # print('b1:{b1}\nb2: {b2}'.format(b1=b1, b2=b2))

        a1 = np.dot(x, W1) + b1
        z1 = self.h1(a1)  # z1 is the cipher
        print(
            # "The dimension of a1 is: {}\n"
            # "And a1 is: {a1}"
            "Cipher(z1) : \n{z1}".format(a1.shape, a1=a1, z1=z1)
        )

        a2 = np.dot(z1, W2) + b2
        z2 = self.h2(W1, a2)  # z2 = np.linalg.inv(W1.dot(W2)).dot(a2) is the Decrypted text

        # print(
        #     "The dimension of a2 is: {d1}\n"
        #     "And a2 is: {a2}\n"
        #     "The dimension of z2 is {d2}\n"
        #     "And z2(Decrypt text) is: {z2}\n".format(d1=a2.shape, a2=a2, d2=z2.shape, z2=z2)
        # )

        return z2

    def h1(self, x):  # activation function 1
        return x

    def h2(self, W, a):  # W is the key1 and a is the output form last layer
        return a.dot(np.linalg.inv(W.dot(1 / W.T)))  # have to use a to multiply the inverse matrix of the identity matrix


# every possible symbol that can be encrypted
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 26 English alphabets
# LETTERS = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'

# message = np.array([*string.ascii_lowercase])[3]  #  array(['d'], dtype='<U1')
Inputs = "a"  # ord('a')->97,  chr(97)->a
# capitalize the string in message
# Inputs = Inputs.upper()
numbers = [ord(letter) for letter in Inputs]

message = np.array(numbers)   # eg: np.array([[1, 2]]), np.array([[[1, 2, 3]]]),
print("Input Plaintext {}\nAnd the dimension is: {}".format(Inputs, message.shape))

# tells the program to encrypt or decrypt
mode = "encrypt"  # set to 'encrypt' or 'decrypt'


# initialize the neural network
input_size, hidden_size, output_size = message.ndim, Bits, message.ndim
test: OneLayerNet = OneLayerNet(input_size, hidden_size, output_size)

Decrypted_text = test.predict(message).astype(int)
print('Decrypt text : {}'.format(chr(Decrypted_text.__int__())))


# run the encryption/decryption code on each symbol in the message string
# stores the encrypted/decrypted form of the message
translated = ""

for symbol in Inputs:
    if symbol in LETTERS:
        # get the encrypted (or decrypted) number for this symbol
        num = LETTERS.find(symbol)  # get the number of the symbol
        if mode == 'encrypt':
            Decrypted_text = test.predict(message).astype(int)
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
print("The cracking code is {}".format(translated))

# copy the encrypted/decrypted string to the clipboard
pyperclip.copy(translated)

