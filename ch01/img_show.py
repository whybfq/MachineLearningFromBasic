# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread
import codecs

img = imread('../dataset/lena.png') #读入图像
plt.imshow(img)

plt.show()