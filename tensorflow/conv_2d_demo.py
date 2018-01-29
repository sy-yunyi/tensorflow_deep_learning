import tensorflow as tf
import numpy as np

minput = np.arange(0,16,1).reshape(1,4,4,1).astype(np.float32)

mfilter = tf.Variable(np.array([1.,2.,3.,4.]).astype(np.float32).reshape([2,2,1,1]))

mstrides = [1,1,1,1]

mpadding = 'SAME'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    conv = tf.nn.conv2d(input=minput,filter=mfilter,strides=mstrides,padding=mpadding)
    print(sess.run(conv).shape)