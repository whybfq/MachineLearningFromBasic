# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread
import codecs

import sys
print(sys.getdefaultencoding())

img = imread('../dataset/lena.png') #读入图像
content = open("../dataset/test.txt", "r").read()

# filehandle.close()
# if content[:3] == codecs.BOM_UTF8:
#     content = content[3:]
# print(content.decode("utf-8"))

# plt.imshow(img)
#
# plt.show()

