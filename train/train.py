# 矩阵操作

# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
#
# sess = tf.Session()

import sys
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import urllib
import pickle


with open('data_batch_1', 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
    print(dir(dict))
    # print(dict)
    # print(type(dict.pop(b'labels')))
    # print(dict.popitem())
    print(dir(dict.items()))
    print(dict.keys())