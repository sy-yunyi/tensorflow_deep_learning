import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import pdb

sess = tf.Session()
mnist = read_data_sets('../Mnist')
train_xdata = np.array([np.reshape(x,(28,28)) for x in mnist.train.images])
test_data = np.array([np.reshape(x,(28,28)) for x in mnist.test.images])
train_labels = mnist.train.labels
test_labels = mnist.test.labels

batch_size = 1000
learning_rate = 0.001
evaluation_size = 10000
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
target_size = max(train_labels) + 1
num_channels = 1
generations = 5000
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2
max_pool_size2 = 2
fully_connected_size1 = 784
# pdb.set_trace()
x_input_shape = (batch_size,image_width,image_height,num_channels)
x_input = tf.placeholder(shape=x_input_shape,dtype=tf.float32,name='x_input')
y_target = tf.placeholder(shape=(batch_size),dtype=tf.int32,name='y_target')
eval_input_shape = (evaluation_size,image_width,image_height,num_channels)
eval_input = tf.placeholder(shape=eval_input_shape,dtype=tf.float32,name='eval_input')
eval_target = tf.placeholder(shape=(evaluation_size),dtype=tf.int32,name='eval_target')

conv1_weight = tf.Variable(tf.truncated_normal([4,4,num_channels,conv1_features],stddev=0.1,dtype=tf.float32),name = 'conv1_weight')
conv1_bias = tf.Variable(tf.zeros([conv1_features],dtype=tf.float32),name='conv1_bias')
conv2_weight = tf.Variable(tf.truncated_normal([4,4,conv1_features,conv2_features],stddev=0.1,dtype=tf.float32),name='conv2_weight')
conv2_bias = tf.Variable(tf.zeros([conv2_features],dtype=tf.float32),name='conv2_bias')


resulting_width = image_width //(max_pool_size1 * max_pool_size2)
resulting_height = image_height // (max_pool_size1 * max_pool_size2)
full1_input_size = resulting_width * resulting_height * conv2_features
full1_weight = tf.Variable(tf.truncated_normal([full1_input_size,fully_connected_size1],stddev=0.1,dtype=tf.float32),name='full1_weight')
full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1],stddev=0.1,dtype=tf.float32),name='full1_bias')
full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1,target_size],stddev=0.1,dtype=tf.float32),name='full2_weiht')
full2_bias = tf.Variable(tf.truncated_normal([target_size],stddev=0.1,dtype=tf.float32),name='full2_bias')

def my_conv_net(input_data):
    conv1 = tf.nn.conv2d(input_data,conv1_weight,strides=[1,1,1,1],padding = 'SAME',name='conv1')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_bias),name='relu1')
    max_pool1 = tf.nn.max_pool(relu1,ksize=[1,max_pool_size1,max_pool_size1,1],strides=[1,max_pool_size1,max_pool_size1,1],padding='SAME',name='max_pool1')

    conv2 = tf.nn.conv2d(max_pool1,conv2_weight,strides=[1,1,1,1],padding='SAME',name='conv2')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_bias),name='relu2')
    max_pool2 = tf.nn.max_pool(relu2,ksize=[1,max_pool_size2,max_pool_size2,1],strides=[1,max_pool_size2,max_pool_size2,1],padding='SAME',name='max_pool2')

    final_conv_shape = max_pool2.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
    flat_output = tf.reshape(max_pool2,[final_conv_shape[0],final_shape])
    fully_connceted1 = tf.nn.relu(tf.add(tf.matmul(flat_output,full1_weight),full1_bias))
    final_model_output = tf.add(tf.matmul(fully_connceted1,full2_weight),full2_bias)
    return (final_model_output)

model_output = my_conv_net(x_input)
test_model_output = my_conv_net(eval_input)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output,labels=y_target))

prediction = tf.nn.softmax(model_output)
test_prediction = tf.nn.softmax(test_model_output)

def get_accuracy(logits,targets):
    batch_predictions = np.argmax(logits,axis =1)
    num_correct = np.sum(np.equal(batch_predictions,targets))
    return (100 * num_correct / batch_predictions.shape[0])

my_opt = tf.train.AdamOptimizer(learning_rate)
train_step = my_opt.minimize(loss,name='train_step')

sess.run(tf.global_variables_initializer())

train_loss = []
train_acc = []
test_acc = []
for i in range(generations):
    rand_index = np.random.choice(len(train_xdata),size=batch_size)
    rand_x = train_xdata[rand_index]
    rand_x = np.expand_dims(rand_x,3)
    rand_y = train_labels[rand_index]
    train_dict = {x_input:rand_x,y_target:rand_y}
    sess.run(train_step, feed_dict=train_dict)
    temp_train_loss,temp_train_preds = sess.run([loss,prediction],feed_dict=train_dict)
    temp_train_acc = get_accuracy(temp_train_preds,rand_y)
    if (i+1)%eval_every == 0:
        eval_index = np.random.choice(len(test_data),size=evaluation_size)
        eval_x = test_data[eval_index]
        eval_x = np.expand_dims(eval_x,3)
        eval_y = test_labels[eval_index]
        test_dict = {eval_input:eval_x,eval_target:eval_y}
        test_preds = sess.run(test_prediction,feed_dict=test_dict)
        temp_test_acc = get_accuracy(test_preds,eval_y)
        train_loss.append(temp_train_preds)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        print(temp_test_acc)


