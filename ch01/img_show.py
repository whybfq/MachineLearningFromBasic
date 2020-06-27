# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread
import codecs
import cProfile

import sys
# print(sys.getdefaultencoding())


def main():
    img = imread('../dataset/lena.png') #读入图像
    plt.imshow(img)
    plt.show()


# content = open("../dataset/test.txt", "r").read()
# filehandle.close()
# if content[:3] == codecs.BOM_UTF8:
#     content = content[3:]
# print(content.decode("utf-8"))


if __name__ == "__main__":
    cProfile.run("main()")

