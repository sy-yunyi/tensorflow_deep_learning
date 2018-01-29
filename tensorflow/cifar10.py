import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt


#超参
train_step = 1000
learning_rate = 0.00006
# num_sample = len(x)
image_width = 32
image_height = 32
num_channels = 3
conv_feature1 = 24
conv_feature2 = 36
fully_connected_num = 1024
batch_size =1000
#加载数据
def loadData(filePath):
    dataSet = []
    labelSet = []
    fileName = 'data_batch_'
    for i in range(1,6):
        fr = open(filePath+'\\'+fileName+str(i),'rb')
        dict = pickle.load(fr,encoding='bytes')
        dataSet.extend(dict.pop(b'data'))
        labelSet.extend(dict.pop(b'labels'))
    dataSet = np.array(dataSet)
    labelSet = np.array(labelSet)
    dataSetTest = pickle.load(open(filePath+'\\'+'test_batch','rb'),encoding='bytes').pop(b'data')
    labelSetTest = pickle.load(open(filePath+'\\'+'test_batch','rb'),encoding='bytes').pop(b'labels')
    dataSet = dataSet.reshape(len(dataSet),3,32,32).transpose(0,2,3,1)
    dataSetTest = dataSetTest.reshape(len(dataSetTest),3,32,32).transpose(0,2,3,1)

    return dataSet,labelSet,dataSetTest,labelSetTest


def cope_img(img):
    # img = tf.image.resize_image_with_crop_or_pad(img,24,24)
    images = []
    for i in range(img.shape[0]):
        final_image = tf.image.random_flip_left_right(img[i])
        final_image = tf.image.random_brightness(final_image, max_delta=63)
        final_image = tf.image.random_contrast(final_image, lower=0.2, upper=1.8)
        images.append(final_image.eval())
    images = np.array(images)
    print(images.shape)
    return images


#生成卷积核
def weight_variable(shape,name,dtype,stddev):
    # return tf.Variable(tf.truncated_normal(shape=shape,stddev=stddev),dtype=dtype,name=name)
    initial = tf.truncated_normal(shape, stddev = stddev)
    return tf.Variable(initial)
#偏置
def bias_variable(shape,name,dtype):
    # return tf.constant(0.1,shape=shape,name=name,dtype=dtype)
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

#卷积，步长为：strides:[1,1,1,1]
def conv_2x2(x,f):
    return tf.nn.conv2d(x,f,strides=[1,1,1,1],padding='SAME')

#最大池化 [1,2,2,1]
def pool_graph(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#cnn model
def cnn_model(x):

    print(x.shape)
    # 第一次卷积
    with tf.variable_scope('conv1') as scope:
        conv1_kernel = weight_variable(shape=[5,5,3,conv_feature1],name='conv_feature1',dtype=tf.float32,stddev=0.1)
        conv1 = conv_2x2(x,conv1_kernel)
        conv_bias1 = bias_variable(shape=[conv_feature1],name='conv_bias1',dtype=tf.float32)
        conv_add_bias1 = tf.nn.bias_add(conv1,conv_bias1)
        relu1 = tf.nn.relu(conv_add_bias1)

    pool1 = pool_graph(relu1)
    #第二次卷积
    with tf.variable_scope('conv2') as scope:
        conv2_kernel = weight_variable(shape=[5,5,conv_feature1,conv_feature2],name='conv_feature2',dtype=tf.float32,stddev=0.1)
        conv2 = conv_2x2(pool1,conv2_kernel)
        conv_bias2 = bias_variable(shape=[conv_feature2],name='conv_bias2',dtype=tf.float32)
        conv_add_bias2 = tf.nn.bias_add(conv2,conv_bias2)
        relu2 = tf.nn.relu(conv_add_bias2)
    pool2 = pool_graph(relu2)

    result_width = image_width // 4
    result_heisht = image_height // 4
    full1_input_size = result_width * result_heisht *conv_feature2
    input_x = tf.reshape(pool2,[-1,full1_input_size])

    #全连接1
    with tf.variable_scope('full1') as scope:
        full1_weight = weight_variable(shape=[full1_input_size,fully_connected_num],stddev=0.1,dtype=tf.float32,name='full1_weight')
        full1_bias = bias_variable(shape=[fully_connected_num],dtype=tf.float32,name='full1_bias')
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(input_x,full1_weight),full1_bias))
    #全连接2
    with tf.variable_scope('full2') as scope:
        full2_weight = weight_variable(shape=[fully_connected_num,10],stddev=0.1,dtype=tf.float32,name='full2_weight')
        full2_bias = bias_variable(shape=[10],name='full2_bias',dtype=tf.float32)
        full2_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1,full2_weight),full2_bias))
    return full2_layer2



# 训练
def cnn_cifar10(x,labels,test,label_test):

    #占位
    input_shape=[None,image_width,image_height,num_channels]

    xPlace = tf.placeholder(shape=input_shape,dtype=tf.float32,name='xPlace')
    labelPlace = tf.placeholder(shape=[None],dtype=tf.int64,name='lablePlace')

    pred = cnn_model(xPlace)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,labels=labelPlace))

    myOpt = tf.train.GradientDescentOptimizer(learning_rate)
    trainProcess = myOpt.minimize(loss,name='trainProcess')

    accur = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),labelPlace),tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(train_step):
            rand_index = np.random.choice(len(x),batch_size)
            randx = x[rand_index,:,:,:]
            randLabel = labels[rand_index]
            # randx = cope_img(randx)
            # print(randx)
            sess.run(trainProcess,feed_dict={xPlace:randx,labelPlace:randLabel})
            tmp = sess.run(accur,feed_dict={xPlace:test,labelPlace:label_test})
            print(tmp)

if __name__ == '__main__':
    filePath = r'C:\Users\Administrator\Desktop\deeplearning\code\SampleData\cifar-10-batches-py'

    dataSet, labelSet, dataSetTest, labelSetTest = loadData(filePath)
    # dataSet = cope_img(dataSet)
    cnn_cifar10(dataSet, labelSet, dataSetTest, labelSetTest)