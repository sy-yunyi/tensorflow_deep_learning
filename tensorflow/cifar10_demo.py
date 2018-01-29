import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
def loadData(filePath):
    data_set = []
    labels_set = []
    fileName = 'data_batch_'
    for i in range(1,6):
        fr = open(filePath+'\\'+fileName+str(i),'rb')
        dict = pickle.load(fr,encoding='bytes')
        data_set.extend(dict.pop(b'data'))
        labels_set.extend(dict.pop(b'labels'))
    fr = open(filePath+'\\'+'test_batch','rb')
    dict = pickle.load(fr,encoding='bytes')
    data_set_test = dict.pop(b'data')

    labels_set_test = dict.pop(b'labels')
    data_set = np.array(data_set)
    print(data_set.shape)
    data_set = data_set.reshape(len(data_set),3,32,32).transpose(0,2,3,1)
    data_set_test = data_set_test.reshape(len(data_set_test),3,32,32).transpose(0,2,3,1)

    # data_set = data_set.reshape(len(data_set),3,32,32)
    # data_set_test = data_set_test.reshape(len(data_set_test),3,32,32)
    #
    # data_set_new = []
    # data_set_test_new = []
    # for i in range(len(data_set)):
    #     data_set_new.append(np.transpose(data_set[i],(1,2,0)))
    # for j in range(len(data_set_test)):
    #     data_set_test_new.append(np.transpose(data_set_test[j],(1,2,0)))

    # plt.imshow(data_set_test[9].reshape(32,32,3))
    # plt.show()

    return np.array(data_set).astype(np.float32),np.array(labels_set).astype(np.float32),np.array(data_set_test).astype(np.float32),np.array(labels_set_test).astype(np.float32)
    # return data_set_new,labels_set,data_set_test_new,labels_set_test
batch_size = 100
learning_rate = 0.00006
image_height = 24
image_width =24
target_size = 10
num_channels = 3
train_step = 20000
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2
max_pool_size2 =2
fully_connected_size = 1024

def weight_variable(shape,name,dtype,stddev = 0.1):
    return (
    tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)))

    # return (tf.Variable(tf.truncated_normal(shape=shape,dtype=dtype,stddev=stddev),name=name))
def bias_variable(shape,name,dtype):
    return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

    # return (tf.Variable(tf.constant(1,shape=shape,dtype=dtype),name=name))

def conv_2x2(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def cnn_model(x):
    sample = tf.reshape(x,[-1,32,32,3])
    with tf.variable_scope('conv1') as scope:
        conv1_kernel = weight_variable(shape=[4,4,3,conv1_features],name='conv1_kernel',dtype=tf.float32)
        conv1_bias = bias_variable(shape=[conv1_features],name='conv1_bias',dtype=tf.float32)
        conv1 = conv_2x2(sample,conv1_kernel)
        conv1_add_bias = tf.nn.bias_add(conv1,conv1_bias)
        relu1 = tf.nn.relu(conv1_add_bias)

    pool1 = tf.nn.max_pool(relu1,ksize=[1,max_pool_size1,max_pool_size1,1],strides=[1,2,2,1],padding='SAME',name='pool1')
    norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm1')
    with tf.variable_scope('conv2') as scope:
        conv2_kernel = weight_variable(shape=[3,3,conv1_features,conv2_features],name='conv2_kernel',dtype=tf.float32)
        conv2_bias = bias_variable(shape=[conv2_features],name='conv2_bias',dtype=tf.float32)
        conv2 = conv_2x2(norm1,conv2_kernel)
        conv2_add_bias = tf.nn.bias_add(conv2,conv2_bias)
        relu2 = tf.nn.relu(conv2_add_bias)

    pool2 = tf.nn.max_pool(relu2,ksize=[1,max_pool_size2,max_pool_size2,1],strides=[1,2,2,1],padding='SAME',name='pool2')
    norm2 = tf.nn.lrn(pool2,depth_radius=5,bias=2.0,alpha=1e-3,beta=0.75,name='norm2')

    resulting_width = image_width //(max_pool_size1 * max_pool_size2)
    resulting_height = image_height // (max_pool_size1 * max_pool_size2)
    full1_input_size = resulting_width * resulting_height * conv2_features
    # print(norm2.get_shape().as_list())
    norm2 = tf.reshape(norm2,[-1,full1_input_size])

    with tf.variable_scope('full1') as scope:
        full1_weight = weight_variable(shape=[full1_input_size,fully_connected_size],dtype=tf.float32,name='full1_weight')
        full_bias1 = bias_variable(shape=[fully_connected_size],dtype=tf.float32,name='full_bias1')
        full_layer1 = tf.add(tf.matmul(norm2,full1_weight),full_bias1)
        relu_layer1 = tf.nn.relu(full_layer1)

    with tf.variable_scope('full2') as scope :
        full2_weight = weight_variable(shape=[fully_connected_size, target_size], dtype=tf.float32,
                                       name='full2_weight')
        full_bias2 = bias_variable(shape=[target_size], dtype=tf.float32, name='full_bias2')
        full_layer2 = tf.add(tf.matmul(relu_layer1, full2_weight), full_bias2)
        relu_result = tf.nn.relu(full_layer2)
    return relu_result

def cifar10_cnn(data_set,labels,test_data_set,lables_test):

    x_input_shape = [None,image_height,image_width,num_channels]
    xPlace = tf.placeholder(shape=x_input_shape,dtype=tf.float32,name='xPlace')
    labelPlace = tf.placeholder(shape=[None],dtype=tf.int64,name='labelPlace')

    pred = tf.nn.softmax(cnn_model(xPlace))

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,labels=labelPlace))

    myopt = tf.train.MomentumOptimizer(learning_rate,0.9)
    trainProcess = myopt.minimize(loss,name='trainProcess')
    accur = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),labelPlace),tf.float32))

    trainLoss = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(train_step):
            currXindex= np.random.choice(len(data_set),size=batch_size)
            currX = data_set[currXindex]
            currLable = labels[currXindex]
            sess.run(trainProcess,feed_dict={xPlace:currX,labelPlace:currLable})
            trainloss =(sess.run(loss,feed_dict={xPlace:currX,labelPlace:currLable}))
            # if (i+1)%5 == 0:
            tempAcc = sess.run(accur,feed_dict={xPlace:test_data_set,labelPlace:lables_test})
            print('acc:%0.3f%%' % float(tempAcc * 100))
            print(trainloss)

if __name__ == '__main__':
    filePath = r'C:\Users\Administrator\Desktop\deeplearning\code\SampleData\cifar-10-batches-py'
    data_set, labels_set, data_set_test, labels_set_test = loadData(filePath)

    data_set = np.array(data_set)
    labels_set = np.array(labels_set)
    data_set_test = np.array(data_set_test)
    labels_set_test = np.array(labels_set_test)

    # cifar10_cnn(data_set,labels_set,data_set_test,labels_set_test)