import tensorflow as tf
import pdb
from numpy.random import RandomState
from PIL import Image
from PIL import PSDraw
import numpy as np
import matplotlib.pyplot as plt

im = Image.open('test.jpg')
# im.show()
print(im.format,im.size,im.mode)
im1 = im.convert('L')
print(im1.format,im1.size,im1.mode)
# im1.show()
# title = 'lena'
# box = (1*72,2*72,7*72,10*72)
# ps= PSDraw.PSDraw()
# ps.begin_document(title)
#
# ps.image(box,im,75)
# im.show()

img = np.array(im)
plt.figure('dog')
plt.show(img)
plt.axis('off')
plt.show()





























#
# g1 = tf.Graph()
# with g1.as_default():
#     v = tf.get_variable('v',shape=[1],initializer=tf.zeros_initializer())
#
# g2 = tf.Graph()
# with g2.as_default():
#     v = tf.get_variable('v',shape=[1],initializer=tf.ones_initializer())
#
# with tf.Session(graph=g1) as sess:
#     tf.initialize_all_variables().run()
#     with tf.variable_scope("",reuse=True):
#         print(sess.run(tf.get_variable('v')))
#
# with tf.Session(graph=g2) as sess:
#     tf.initialize_all_variables().run()
#     with tf.variable_scope("",reuse=True):
#         print(sess.run(tf.get_variable('v')))
#
# sess = tf.Session()
# a = tf.constant([1.0,2.0],name='a')
# b = tf.constant([2.0,3.0],name='b')
# result = tf.add(a,b,name='add')
# with sess.as_default():
#     print(result.eval())
# # sess.run(result)
#
# weights = tf.Variable(tf.random_normal([2,3],stddev=2))
#
# w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
# w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
#
# x = tf.constant([[0.7,0.9]])
#
# a = tf.matmul(x, w1)
#
# y = tf.matmul(a, w2)
#
# sess = tf.Session()
# init_op = tf.initialize_all_variables()
# sess.run(init_op)
# sess.run(w1.initializer)
# sess.run(w2.initializer)
# pdb.set_trace()
# print(sess.run(y))
# sess.close()

# batch_size = 8
#
# w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed = 1))
# w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
#
# x = tf.placeholder(tf.float32,shape=(None,2),name='x_input')
# y_ = tf.placeholder(tf.float32,shape=(None,1),name='y_input')
#
# a = tf.matmul(x,w1)
# y = tf.matmul(a,w2)
#
# cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
# train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#
# rdm = RandomState(1)
# data_size = 128
# X= rdm.rand(data_size, 2)
# Y = [[int(x1+x2<1)] for (x1,x2) in X]
#
# with tf.Session() as sess:
#     init_op = tf.initialize_all_variables()
#     sess.run(init_op)
#     print(sess.run(w1))
#     print(sess.run(w2))
#     steps = 5000
#     for i in range(steps):
#         start = (i * batch_size) % data_size
#         end = min(start+batch_size,data_size)
#         sess.run(train_step,feed_dict={x:X[start:end],y_:Y})
