import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# x_vals = np.random.normal(1,0.1,100)
# y_vals = np.repeat(10.,100)
# y = [x * 10 for x in x_vals]
# plt.scatter(x_vals,y_vals)
# plt.show()

x_vals = np.linspace(0,10,100)
y_vals = x_vals + np.random.normal(0,1,100)
# xn_vals =np.mat(x_vals.reshape([1,100]))
x_tf = tf.convert_to_tensor(x_vals)

train_index = np.random.choice(len(x_vals),round(len(x_vals)*0.8),replace=False)
test_index = np.array(list(set(range(len(x_vals)))-set(train_index)))

train_x = x_vals[train_index]
train_y = y_vals[train_index]

test_x = x_vals[test_index]
test_y = y_vals[test_index]
plt.plot(train_x,train_y,'r+')
plt.plot(test_x,test_y,'g+')
plt.show()


# 占位：数据入口
x_data = tf.placeholder(shape=[None],dtype=tf.float32,name='x_data')
y_target = tf.placeholder(shape=[None],dtype=tf.float32,name='y_target')

# 参数 A,目标曲线my_output
A = tf.Variable(tf.random_normal(shape=[],name='A'))
b = tf.Variable(tf.random_normal(shape=[],name='b'))
my_output = tf.add(tf.multiply(A,x_data,name='my_output1'),b,name='my_output')

# 损失函数
loss = tf.reduce_mean(tf.square(my_output - y_target))

# tf.summary.scalar('loss',loss)
# merged = tf.summary.merge_all()

# 定义优化器
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss, name='train_step')

losses_train = []
losses_test = []


# w = tf.matrix_inverse( tf.transpose(x_tf) * x_tf)



# print(w1)
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # my_graph = tf.summary.FileWriter('E:\\tensorboardlog\\',sess.graph)
    print(type(x_tf))
    for i in range(500):
        # rand_index = np.random.choice(100,50,replace=False)
        # rand_x = x_vals[rand_index]
        # rand_y = y_vals[rand_index]
        # my_graph = tf.summary.FileWriter('E:\\tensorboardlog\\',sess.graph)
        sess.run(train_step,feed_dict={x_data:train_x,y_target:train_y})
        # my_graph.add_summary(summary,i)
        losses_train.append(sess.run(loss,feed_dict={x_data:train_x,y_target:train_y}))
        losses_test.append(sess.run(loss,feed_dict={x_data:test_x,y_target:test_y}))

    A =sess.run(A)
    # print(type(A))
    plt.plot([i for i in range(500)],losses_train,'r')

    plt.plot([i for i in range(500)],losses_test,'g')
    plt.show()
    #
    x = np.linspace(0,10,100)
    y = [A * x for x in x_vals]
    # print(A)
    test_x = x_vals[test_index]
    test_y = y_vals[test_index]
    plt.plot(train_x, train_y, 'r+')
    plt.plot(test_x, test_y, 'g+')
    plt.plot(x,y)
    # plt.show()


X = np.column_stack((x_vals.reshape(-1,1),np.ones(100).reshape(-1,1)))
# print(X)
Y = y_vals.reshape(-1,1)
print(Y)

solution = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(X),X)),X),Y)
with tf.Session() as sess:
    result = sess.run(solution)
    print(result)


