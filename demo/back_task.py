import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 数据
x_vals = np.concatenate((np.random.normal(-3,1,50),np.random.normal(3,1,50)))
y_vals = np.concatenate((np.repeat(0.,50),np.repeat(1.,50)))

# plt.scatter(x_vals,y_vals)
# plt.show()
x_data = tf.placeholder(shape=[None],name='x_data',dtype=tf.float32)
y_target = tf.placeholder(shape=[None],name='y_target',dtype=tf.float32)

A = tf.Variable(tf.random_normal(mean=10,shape=[]))

# b = tf.Variable(tf.random_normal(shape=[]))

my_output = tf.add(x_data,A,name='my_output1')
# my_output = tf.add(my_output1, b,name='my_output')
# my_output = y_target-my_output2
# loss = -tf.reduce_mean(y_vals * tf.log(my_output + 1e-10))
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output ,labels=y_target))

my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss,name='train_step')

losses = []

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        rand_index = np.random.choice(100,80,replace=False)
        rand_x = x_vals[rand_index]
        rand_y = y_vals[rand_index]
        sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
        losses.append(sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y}))
        # print(A.eval())
    # print(losses)
    plt.plot([i for i in range(1000)],losses)
    plt.show()