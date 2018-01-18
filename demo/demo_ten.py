import tensorflow as tf
import numpy as np

# 占位，
# x = tf.placeholder(shape=[],dtype=tf.int32,name='x')
#
# w = tf.Variable(tf.constant(10),name='w')
# w = w.assign(100)
# b = tf.Variable(tf.constant(5),name='b')
#
# y = w * x +b
#
# x_data = [i for i in range(3)]
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for item in x_data:
#         print(sess.run(y,feed_dict={x:item}))

x_vals = tf.random_normal(1,0.1,100)
y_vals = np.repeat(10.,100)

# 占位：
x_data = tf.placeholder(shape=[None],dtype=tf.float32,name='x_data')
y_target = tf.placeholder(shape=[None],dtype=tf.float32,name='y_target')

A = tf.Variable(tf.random_normal(shape=[],name='A'))
my_output = tf.multiply(x_data,A,name='my_output')

loss = tf.square(my_output - y_target)

my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss,name='train_step')

losses = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        rand_index = np.random.choice(100,10,replace=False)
        rand_x = x_vals[rand_index]
        rand_y = y_vals[rand_index]
        sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
        losses.append(sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y}))
