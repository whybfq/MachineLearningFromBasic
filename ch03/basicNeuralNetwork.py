# coding: utf-8
"""
basicNeuralNetwork.py (BSD Licensed) : only use one layer
© 2020 (littlesanner@gmail.com)
version: 0.0.1
"""

import time
import numpy as np
import pyperclip
import sys
import os

# every possible symbol that can be encrypted
# LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # if only consider 26 English alphabets
# consider more situations, note the space at the front
LETTERS = """ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"""
Bits = 5  # hidden_size
# c = np.pi / 2  # consider to use sin()/cos() to b, need to test and design


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


def makeKeyFiles(filename: str, matrix: np) -> np:
    """
    # Creates one file 'name' with the matrix written in it
    # Our safety check will prevent us from overwriting our old key files:
    # write name (eg keyA.txt)
    :arg name: filename
    """
    # if os.path.exists(filename):
    #     print(f"WARNING: The file {filename} exists!  "
    #           "Use a different name or delete these files and re-run this program.")
    #     sys.exit()
    print()
    if type(matrix) is np.ndarray:  # if isinstance(matrix, np.ndarray)
        print(f"The matrix's dimension is {matrix.shape} ")
    print(f'Writing to file {filename}...')
    fo = open(f'{filename}', 'w')
    fo.write(f'{matrix}')  # the content of "name"(file)
    fo.close()


def compareFiles(file1, file2):
    """
    :type file1: str
    :type file2: str
    :return: some message
    """
    data1, data2 = open(file1, 'r').read(), open(file2, 'r').read()
    if data1 == data2:
        print(f"The {file1} and {file2} are same!")
    else:
        print(f"The {file1} and {file2} are different!")


class OneLayerNet:  # including encryptMessage() and decryptMessage()

    # def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std=0.01):
    #     # initialize the weight and bias
    #     self.params = dict(
    #         W1=weight_init_std * np.random.randn(input_size, hidden_size),
    #         b1=np.zeros(hidden_size),
    #         b2=np.zeros(output_size)  # the size if the same as input_size
    #     )

    def h1(self, x):  # activation function 1
        return x

    def h2(self, a, W):  # W is the KeyA and a is the output form last layer
        return a.dot(
            np.linalg.inv(W.dot(1 / W.T))
        )  # have to use a to multiply the inverse matrix of the identity matrix

    def encrypted(self, message):
        input_size, hidden_size = message.ndim, Bits  # if only encrypted one bit then input_size=1
        weight_init_std = 0.01
        W1 = weight_init_std * np.random.randn(input_size, hidden_size)
        b1 = np.zeros(hidden_size)

        a1 = np.dot(message, W1) + b1
        z1 = self.h1(a1)  # z1 is the cipher
        return z1, W1  # return key and cipher_text

    def decrypted(self, z1, W1):
        output_size = 1  # output_size = message.ndim
        b2 = np.zeros(output_size)  # the size if the same as input_size
        a2 = np.dot(z1, (1 / W1.T)) + b2
        z2 = self.h2(a2, W1)  # z2 = np.linalg.inv(W1.dot(W2)).dot(a2) is the Decrypted text
        return z2

    def predict(self, x):
        W1, b1, b2 = self.params['W1'], self.params['b1'], self.params['b2']
        W2 = 1 / W1.T  # W2 is from W1
        # print(
        #     "The dimension of W1 is {d1}\n"
        #     "And W1(keyA) is: {W1}\n"
        #     "The dimension of W2 is {d2}\n"
        #     "And W2(keyB) is: {W2}".format(d1=W1.shape, W1=W1, d2=W2.shape, W2=W2)
        # )
        # print(f'b1:{b1}\nb2: {b2}')

        # makeKeyFiles("keyA.txt", W1)  # write keyA.txt
        # makeKeyFiles("KeyB.txt", W2)  # write keyB.txt

        a1 = np.dot(x, W1) + b1
        z1 = self.h1(a1)  # z1 is the cipher
        # print(
        #     # "The dimension of a1 is: {}\n"
        #     # "And a1 is: {a1}"
        #     "Cipher text(z1) : \n{z1}".format(a1.shape, a1=a1, z1=z1)
        # )

        # makeKeyFiles("Cipher.txt", z1)  # write Cipher.txt

        a2 = np.dot(z1, W2) + b2
        z2 = self.h2(a2, W1)  # z2 = np.linalg.inv(W1.dot(W2)).dot(a2) is the Decrypted text
        # print(
        #     "The dimension of a2 is: {d1}\n"
        #     "And a2 is: {a2}\n"
        #     "The dimension of z2 is {d2}\n"
        #     "And z2(Decrypt text) is: {z2}\n".format(d1=a2.shape, a2=a2, d2=z2.shape, z2=z2)
        # )

        # makeKeyFiles("KeyB.txt", W2)  # write keyB.txt
        # makeKeyFiles("Decrypted.txt", z2)  # write Decrypted.txt

        return z2

    def checkKeys(self, keyA, keyB, mode):
        keyA = self.params['W1'] + self.params['b1']
        keyB = 1 / keyA.T + self.params['b1']
        if keyA == 1 and mode == 'encrypt':
            sys.exit('The affine cipher becomes incredibly weak when key A is set to 1. Choose a different key.')
        if keyB == 0 and mode == 'encrypt':
            sys.exit('The affine cipher becomes incredibly weak when key B is set to 0. Choose a different key.')
        # if keyA < 0 or keyB < 0 or keyB > len(LETTERS) - 1:
        #     sys.exit('Key A must be greater than 0 and Key B must be between 0 and %s.' % (len(LETTERS) - 1))


@calculate_time
def main():
    myMessage = open("Input.txt").read()
    # myMessage = """"A computer would deserve to be called intelligent if it could deceive a human into believing that it was human." -Alan Turing"""
    # Mode = "encrypt"  # set to 'encrypt' or 'decrypt'
    Mode = "decrypt"  # set to 'encrypt' or 'decrypt'

    key1, translated = [], []
    if Mode == 'encrypt':
        for symbol in myMessage:
            if symbol in LETTERS:
                message = np.array([ord(symbol)])  # eg: np.array([[1, 2]]), np.array([[[1, 2, 3]]]),

                # initialize the neural network
                test: OneLayerNet = OneLayerNet()
                keyA, Encrypted_text = test.encrypted(message)
                print(f"keyA(W1) is {keyA}\n"
                      f"And its dimension is {keyA.ndim}\n"
                      f"And its shape is {keyA.shape}\n"
                      f"Encrypted_text(z1) is {Encrypted_text}\n"
                      f"And its dimension is {Encrypted_text.ndim}\n"
                      f"And z1's shape is {Encrypted_text.shape}")

                for i in keyA:  # key.extend(i for i in keyA)
                    key1.append(i)
                for j in Encrypted_text:  # need to be improved to dimensions
                    for i in j:
                        translated.append(i)

            else:  # if symbol not in the LETTERS
                key1.append([])
                translated.append(symbol)

        makeKeyFiles("key1_Ndarray.txt", np.asarray(key1))  # convert the list to a numpy array
        makeKeyFiles("Cipher_Ndarray.txt", np.asarray(translated))
        makeKeyFiles("key1_String.txt", ' '.join([ str(elem) for elem in key1 ]))  # convert the list to string
        makeKeyFiles("Cipher_String.txt", ' '.join(map(str, translated)))  # convert the list to string
        makeKeyFiles("key1_list.txt", key1)
        makeKeyFiles("Cipher_list.txt", translated)

    elif Mode == 'decrypt':
        test: OneLayerNet = OneLayerNet()
        # after open the file, remove the "[" and "]" in the beginning and end of the string
        cipher_text, key = open("key1_list.txt", 'r').read().replace('[', '').replace(']', ''), \
                           open("Cipher_list.txt", 'r').read().replace('[', '').replace(']', '')
        for i in cipher_text.split(", "):
            translated.append(float(i))  # float() to convert string to list
        for j in key.split(", "):
            key1.append(float(j))

        print(f"You input :\nz1(cipher) is {translated}\nW1(key) is {key1}")

        # cipher = np.array([0.4309343925444561, -0.40563050290909614, 0.7161005261005042])  # Bits = 3
        # key = np.array([[0.0056701893755849485, -0.005337243459330212, 0.009422375343427688]])
        # print(f"Cipher text: {cipher[ 0:Bits ]}\nKey is: {key[ 0:Bits ]}")
        i, plaintext = 0, ""
        # main process to decrypt each every character
        while i < len(key1):  # len(key1) is equal to len(translated)
            cipher = np.array(translated[i:i + Bits])
            key = np.array([key1[i: i + Bits]])
            plaintext += chr(test.decrypted(z1=cipher, W1=key).__int__())  # convert numpy.array to int then to chr()
            i += Bits
        print(f"The plaintext is {plaintext}")
        makeKeyFiles("Decrypted.txt", plaintext)
        compareFiles("Input.txt", "Decrypted.txt")
        translated = plaintext

    else:
        print('mode can only be encrypt or decrypt!')

    print(f'Key1: {key1}')  # key1 will be W1
    # if mode == 'encrypt', translated will be z1(cipher text)
    # if mode == 'decrypt', translated will be z2(plaintext)
    print(f'Translated text: {translated}')
    # copy the encrypted/decrypted string to the clipboard
    # print("already copied to the clipboard of translated!")
    # pyperclip.copy(translated)


if __name__ == '__main__':
    main()
