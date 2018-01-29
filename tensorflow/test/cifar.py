import os
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python.framework import ops
#参数
batch_size = 100
learnin_rate = 0.02
train_size = 1

#加载cifar数据集函数
def load_cifar_batch(fileName):
	with open(fileName,'rb') as fr:
		datadict = pickle.load(fr,encoding='latin1')
		x_val = datadict['data'].reshape((10000,3,32,32)).transpose((0,2,3,1))
		y_val = datadict['labels']
	return x_val,y_val

def load_cifar(folder):
	xs = []
	ys = []
	for b in range(1,6):
		fname  = os.path.join(folder, 'data_batch_%d'%b)
		x_val, y_val = load_cifar_batch(fname)
		xs.append(x_val)
		ys.append(y_val)
	x_vals_train = np.concatenate(xs)
	y_vals_train = np.concatenate(ys)
	x_vals_test, y_vals_test = load_cifar_batch(os.path.join(folder,'test_batch'))
	return x_vals_train,y_vals_train,x_vals_test,y_vals_test
#初始化权重和偏置
def init_weight(shape, std_dev):
	weight = tf.Variable(tf.truncated_normal(shape, stddev=std_dev))
	return weight
def init_bias(shape, std_dev):
	bias = tf.Variable(tf.truncated_normal(shape, stddev=std_dev))
	return bias
#卷积
def conv2d(x,w):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1], padding='SAME')
#池化
def max_pool_3x3(x):
	return tf.nn.max_pool(x,ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
#全链接
def fully_connected(input_layer, weights, biases):
	layer = tf.add(tf.matmul(input_layer,weights), biases)
	return tf.nn.relu(layer)

#加载数据集	
x_vals_train,y_vals_train,x_vals_test,y_vals_test = load_cifar('cifar-10-batches-py')
#声明占位符
x_data = tf.placeholder(shape=[None,32,32,3], dtype=tf.float32)
y_target= tf.placeholder(shape=[None], dtype=tf.int64)
#算法模型
#第一层卷积
weight_1 = init_weight([5,5,3,64], 0.05)
bias_1 = init_bias([64], 0.05)
layer1 = max_pool_3x3(tf.nn.relu(tf.nn.bias_add(conv2d(x_data,weight_1),bias_1)))
#第二层卷积
weight_2 = init_weight([5,5,64,64], 0.05)
bias_2 = init_bias([64], 0.05)
layer2 = max_pool_3x3(tf.nn.relu(tf.nn.bias_add(conv2d(layer1,weight_2),bias_2)))
#第三层 连接
reshaped_output = tf.reshape(layer2, [-1,8*8*64])
weight_3 = init_weight(shape=[8*8*64,384],std_dev=0.05)
bias_3 = init_bias(shape=[384],std_dev=0.05)
layer3 = fully_connected(reshaped_output,weight_3,bias_3)
#第四层 连接
weight_4 = weight_4 = init_weight(shape=[384,192],std_dev=0.05)
bias_4 = init_bias(shape=[192],std_dev=0.05)
layer4 = fully_connected(layer3,weight_4,bias_4)
#最终输出层
weight_5 = init_weight(shape=[192,10],std_dev=0.05)
bias_5 = init_bias(shape=[10],std_dev=0.05)
final_layer = tf.nn.softmax(tf.add(tf.matmul(layer4,weight_5), bias_5))
#损失函数和优化器
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target, logits=final_layer)
train_step = tf.train.AdamOptimizer(learnin_rate).minimize(loss)
#准确率
accury = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(final_layer,1),y_target), dtype=tf.float32))
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(train_size):
		for j in range(500):
			rand_index = np.random.choice(50000,batch_size)
			rand_x = x_vals_train[rand_index]
			rand_y = y_vals_train[rand_index]
			sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})
		accur = sess.run(accury, feed_dict={x_data:x_vals_test, y_target:y_vals_test})
		print(accur)




