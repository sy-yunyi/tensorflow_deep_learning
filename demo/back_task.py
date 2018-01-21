import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pdb
# 数据
x_vals = np.concatenate((np.random.normal(-3,1,50),np.random.normal(3,1,50)))
y_vals = np.concatenate((np.random.normal(-2,1,50),np.random.normal(2,1,50)))
y_vals = np.concatenate((np.repeat(0.,50),np.repeat(1.,50)))
# plt.scatter(x_vals,y_vals)
# plt.show()
train_index = np.random.choice(len(x_vals),round(len(x_vals)*0.8),replace=False)
test_index = np.array(list(set(range(len(x_vals)))-set(train_index)))

train_x = x_vals[train_index]
train_y = y_vals[train_index]

test_x = x_vals[test_index]
test_y = y_vals[test_index]
# plt.plot(train_x,train_y,'r+')
# plt.plot(test_x,test_y,'g+')
# plt.show()


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

current_p = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.nn.sigmoid(my_output)),y_target),tf.float32))

my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss,name='train_step')


losses_data =[]

losses = []
step =0
data_size =[1,20,100]
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
    # 随机样本
        # rand_index = np.random.choice(100)
        # 随机批次
        rand_index = np.random.choice(80,30,replace=False)
        rand_x = train_x[rand_index]
        rand_y = train_y[rand_index]
        sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
        losses.append(sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y}))
        p =sess.run(current_p,feed_dict={x_data:test_x,y_target:test_y})
        losses_data.append(losses)
        # linex = np.linspace(-5, 5, 1000)
        # a = sess.run(A)
        # if i % 500 == 0:
        #     plt.plot(linex, linex * a , label = (i/500 +1))
    print('正确率为：%0.2f%%' % (p*100))
    # linex = np.linspace(-5, 5, 1000)
    # for i in range(len(losses_data)):
    #     plt.plot(linex, losses_data[i])
    fig = plt.figure()
    a = sess.run(A)
    index = np.linspace(start=-5,stop=5,num=100)
    plt.scatter(test_x,test_y,c = 'g',marker='>',s=20)
    plt.scatter(train_x,train_y,c = 'r',marker='*',s=10)
    plt.plot(index,index + a)
    plt.show()
    linex = np.linspace(-5, 5, 1000)
    plt.plot(linex,losses)
    plt.show()