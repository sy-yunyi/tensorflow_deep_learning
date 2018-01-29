import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def weight_variable(shape,name,stddev=0.1,dtype=tf.float32):
    initial = tf.truncated_normal(shape=shape,stddev=stddev,dtype=dtype,name=name)
    return tf.Variable(initial)

def bias_variable(shape,name):
    initial = tf.constant(0.1,shape = shape,name = name)
    return tf.Variable(initial)

def conv2d(x,W):
    pass