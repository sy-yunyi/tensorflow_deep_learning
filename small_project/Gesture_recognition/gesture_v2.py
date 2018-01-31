import numpy as np
import tensorflow as tf
import pickle
import random
import matplotlib.pyplot as plt
import os
from PIL import Image
#超参
#训练参数
batch_size = 30
train_step = 1000

learning_rate = 0.00004

#图片参数
image_height = 64
image_width = 64
num_channels = 3  #通道数

target_size = 10  #目标分类数

#cnn参数
conv_feature1 = 16
conv_feature2 = 64
conv_feature3 = 128

pool_size1 = 2
pool_size2 = 2
pool_size3 = 4

fully_connected_num1 = 512
fully_connected_num2 = 512
fully_connected_num3 = 256


# def readImg():
#     dataSet = []
#     labelSet = []
#     for root,dir,file in os.walk(r'E:\dataset\gesture'):
#         for d in dir:
#             for root, dir, files in os.walk(r'E:\dataset\gesture'+'\\'+d):
#                 for f in files:
#                     labelSet.append(f[0])
#                     img = Image.open(r'E:\dataset\gesture'+'\\'+d+'\\'+f)
#                     img = img.resize((64,64))
#                     im = np.array(img)
#                     dataSet.append(im)
#     print(np.array(dataSet).shape)
#     np.save('gesture2',np.array(dataSet))
#     np.save('gesture_labels2',np.array(labelSet))
#     return np.array(dataSet),np.array(labelSet)

def loadData(filePath,filePath_label):
    dataSet = np.load(filePath)
    labels = np.load(filePath_label).astype(np.int)
    return dataSet,labels



#初始化参数
def init_weight(name,dtype,shape,stddev = 0.001):
    # return tf.Variable(tf.truncated_normal(shape=shape,stddev=stddev),dtype=dtype,name=name)
    return (
    tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)))

#初始化偏置
def init_bias(name,dtype,shape):
    return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

def conv_x(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
def max_pool(x,ksize):
    return tf.nn.max_pool(x,ksize=ksize,strides=[1,2,2,1],padding='SAME')

def cnn_model(x):

    #第一次卷积
    with tf.variable_scope('conv1') as scope:
        conv_kernel1 = init_weight(shape=[5,5,3,conv_feature1],dtype=tf.float32,name='conv_kernel1')
        conv_bias1 = init_bias(shape=[conv_feature1],dtype=tf.float32,name='conv_bias1')
        conv1 = conv_x(x,conv_kernel1)
        conv1_add_bias = tf.add(conv1,conv_bias1)
        relu1 = tf.nn.tanh(conv1_add_bias)

    #第一次池化
    pool1 = max_pool(relu1,ksize=[1,pool_size1,pool_size1,1])
    #局部响应归一化
    norm1 = tf.nn.lrn(pool1)

    #第二次卷积
    with tf.variable_scope('conv2') as scope:
        conv_kernel2 = init_weight(shape=[5,5,conv_feature1,conv_feature2],dtype=tf.float32,name='conv_kernel2')
        conv_bias2 = init_bias(shape=[conv_feature2],dtype=tf.float32,name='conv_bias2')
        conv2 = conv_x(norm1,conv_kernel2)
        conv2_add_bias = tf.add(conv2,conv_bias2)
        relu2 = tf.nn.tanh(conv2_add_bias)

    pool2 = max_pool(relu2,ksize=[1,pool_size2,pool_size2,1])
    norm1 = tf.nn.lrn(pool2)
    result_height = image_height //(pool_size2 * pool_size1)
    result_width = image_width // (pool_size1 * pool_size2)
    full_input_size1 = result_height * result_width * conv_feature2
    # print(full_input_size1,image_height // pool_size2 * pool_size1,result_width)
    input_x = tf.reshape(norm1,[-1,full_input_size1])

    #全连接1
    with tf.variable_scope('full1') as scope:
        full_weight1 = init_weight(shape=[full_input_size1,fully_connected_num1],dtype=tf.float32,name='full_weight1')
        full_bias1 = init_bias(shape=[fully_connected_num1],dtype=tf.float32,name='full_bias1')
        fully_layer1 = tf.add(tf.matmul(input_x,full_weight1),full_bias1)
        relu_layer1 = tf.nn.tanh(fully_layer1)

    # dropout
    # relu_layer1 = tf.nn.dropout(relu_layer1,0.3)

    with tf.variable_scope('full2') as scope :
        full_weight2 = init_weight(shape=[fully_connected_num1,fully_connected_num2],dtype=tf.float32,name='full_weight2')
        full_bias2 = init_bias(shape=[fully_connected_num2],dtype=tf.float32,name='full_bias2')
        fully_layer2 = tf.add(tf.matmul(relu_layer1,full_weight2),full_bias2)
        relu_layer2 = tf.nn.tanh(fully_layer2)

    with tf.variable_scope('full3') as scope:
        full_weight3 = init_weight(shape=[fully_connected_num2,target_size],dtype=tf.float32,name='full_weight3')
        full_bias3 = init_bias(shape=[target_size],dtype=tf.float32,name='full_bias3')
        fully_layer3 = tf.add(tf.matmul(relu_layer2,full_weight3),full_bias3)
        relu_layer3 = tf.nn.tanh(fully_layer3)

    return relu_layer3

def gesture_cnn(x,labels):

    trainPrecent = 0.7
    num_sample = x.shape[0]

    index = [i  for i in range(x.shape[0])]
    random.shuffle(index)
    trainIndex = index[:round(trainPrecent * num_sample)]
    testIndex = index[round(trainPrecent * num_sample) : ]

    trainSample = x[trainIndex]
    trainLabels = labels[trainIndex]
    testSample = x[testIndex]
    testLabels = labels[testIndex]



    x_input_shape = [None,image_width,image_height,num_channels]
    xPlace = tf.placeholder(shape=x_input_shape,dtype=tf.float32,name='xPlace')
    labelPlace = tf.placeholder(shape=[None],dtype=tf.int64,name='labelPlace')

    pred = cnn_model(xPlace)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,labels=labelPlace))

    accur = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),labelPlace),tf.float32))

    myOpt = tf.train.AdamOptimizer(learning_rate)

    trainProcess = myOpt.minimize(loss,name='trainProcess')

    losses_tr = []
    losses_te = []
    acc = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(train_step):
            sess.run(trainProcess,feed_dict={xPlace:trainSample,labelPlace:trainLabels})
            losses_tr.append(sess.run(loss,feed_dict={xPlace:trainSample,labelPlace:trainLabels}))
            losses_te.append(sess.run(loss,feed_dict={xPlace:testSample,labelPlace:testLabels}))
            acc.append(sess.run(accur,feed_dict={xPlace:testSample,labelPlace:testLabels}))
            print('step : {step}   acc = {acc}   loss_te = {loss_te}   loss_tr = {loss_tr}'.format
                  (step = i,acc = acc[i],loss_te = losses_te[i],loss_tr=losses_tr[i]))

        # 模型保存
        saver = tf.train.Saver()
        saver.save(sess, 'model/gesture_v2/')

        linex = np.linspace(-2,8,num = train_step)
        print('last acc : ',acc[-1])
        plt.plot(linex,losses_tr,c='g',label = 'train')
        plt.plot(linex,losses_te,c='r',label = 'test')
        plt.plot(linex,acc,c='b',label = 'accur')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    filePath = 'gesture2.npy'
    filePath_label = 'gesture_labels2.npy'
    # readImg()
    dataSet, labels = loadData(filePath,filePath_label)
    gesture_cnn(dataSet, labels)




