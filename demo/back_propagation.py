import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_vals = np.random.normal(1,0.1,100)
y_vals = np.repeat(10.,100)
# plt.scatter(x_vals,y_vals)
# plt.show()



# 占位：数据入口
x_data = tf.placeholder(shape=[None],dtype=tf.float32,name='x_data')
y_target = tf.placeholder(shape=[None],dtype=tf.float32,name='y_target')

# 参数 A,目标曲线my_output
A = tf.Variable(tf.random_normal(shape=[],name='A'))
my_output = tf.multiply(A,x_data,name='my_output')

# 损失函数
loss = tf.square(my_output - y_target)

# 定义优化器
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss, name='train_step')

losses = []

with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    for i in range(500):
        rand_index = np.random.choice(100)
        rand_x = x_vals[rand_index]
        rand_y = y_vals[rand_index]
        # my_graph = tf.summary.FileWriter('E:\\tensorboardlog\\',sess.graph)
        sess.run(train_step,feed_dict={x_data:[rand_x],y_target:[rand_y]})
        losses.append(sess.run(loss,feed_dict={x_data:[rand_x],y_target:[rand_y]}))

    plt.plot([i for i in range(500)],losses)
    plt.show()




