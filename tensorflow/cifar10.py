import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt


#超参
train_step = 100000
learning_rate = 0.00003
# num_sample = len(x)
image_width = 32
image_height = 32
num_channels = 3
conv_feature1 = 64
conv_feature2 = 64
fully_connected_num1 = 1024
fully_connected_num2 = 512
# keep_prob = 0.5
lr_decay = 0.9
num_gens_to_wait = 250

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

    dataSet = dataSet.reshape(len(dataSet),3,32,32)
    # dataSet = norm_data(dataSet)
    dataSet = dataSet.transpose(0,2,3,1)
    dataSetTest = dataSetTest.reshape(len(dataSetTest),3,32,32)
    # dataSetTest = norm_data(dataSetTest)
    dataSetTest = dataSetTest.transpose(0,2,3,1)

    return dataSet,labelSet,dataSetTest,labelSetTest

# 图片归一化
def norm_data(dataSet):
    # print(dataSet.shape)
    data_mean = np.mean(dataSet,axis=3)
    data_std = np.std(dataSet,axis=3)
    for i in range(len(dataSet)):
        for j in range(3):
            # print(dataSet[i][j])
            dataSet[i][j] = (dataSet[i][j]-data_mean[i][j])/data_std[i][j]
            # print(dataSet[i][j])
    return dataSet

# 图片处理
def cope_img(img):
    # img = tf.image.resize_image_with_crop_or_pad(img,24,24)
    images = []
    for i in range(img.shape[0]):
        # final_image = tf.image.random_flip_left_right(img[i])
        # final_image = tf.image.random_brightness(img[i], max_delta=63)
        # final_image = tf.image.random_contrast(final_image, lower=0.2, upper=1.8)
        final_image = tf.image.per_image_standardization(img[i])
        images.append(final_image.eval())
    images = np.array(images)
    # print(images.shape)
    return images


#生成卷积核
def weight_variable(shape,name,dtype,stddev=0.001):
    # return tf.Variable(tf.truncated_normal(shape=shape,stddev=stddev),dtype=dtype,name=name)
    initial = tf.truncated_normal(shape, stddev = stddev)
    return tf.Variable(initial)
#偏置
def bias_variable(shape,name,dtype):
    # return tf.constant(0.1,shape=shape,name=name,dtype=dtype)
    initial = tf.constant(0.001, shape = shape)
    return tf.Variable(initial)

#卷积，步长为：strides:[1,1,1,1]
def conv_2x2(x,f):
    return tf.nn.conv2d(x,f,strides=[1,1,1,1],padding='SAME')

#最大池化 [1,2,2,1]
def pool_graph(x):
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

#cnn model
def cnn_model(x):

    print(x.shape)
    # 第一次卷积
    with tf.variable_scope('conv1') as scope:
        conv1_kernel = weight_variable(shape=[5,5,3,conv_feature1],name='conv_feature1',dtype=tf.float32)
        conv1 = conv_2x2(x,conv1_kernel)
        conv_bias1 = bias_variable(shape=[conv_feature1],name='conv_bias1',dtype=tf.float32)
        conv_add_bias1 = tf.nn.bias_add(conv1,conv_bias1)
        relu1 = tf.nn.relu(conv_add_bias1)

    pool1 = pool_graph(relu1)
    norm1= tf.nn.lrn(pool1,depth_radius=5,bias=1.0,alpha= 1e-3,beta = 0.75,name='norm1')

    #第二次卷积
    with tf.variable_scope('conv2') as scope:
        conv2_kernel = weight_variable(shape=[5,5,conv_feature1,conv_feature2],name='conv_feature2',dtype=tf.float32)
        conv2 = conv_2x2(norm1,conv2_kernel)
        conv_bias2 = bias_variable(shape=[conv_feature2],name='conv_bias2',dtype=tf.float32)
        conv_add_bias2 = tf.nn.bias_add(conv2,conv_bias2)
        relu2 = tf.nn.relu(conv_add_bias2)
    pool2 = pool_graph(relu2)
    #局部响应归一化
    norm2= tf.nn.lrn(pool2,depth_radius=5,bias=1.0,alpha= 1e-3,beta = 0.75,name='norm1')

    result_width = image_width // 4
    result_heisht = image_height // 4
    full1_input_size = result_width * result_heisht *conv_feature2
    input_x = tf.reshape(norm2,[-1,full1_input_size])

    #全连接1
    with tf.variable_scope('full1') as scope:
        full1_weight = weight_variable(shape=[full1_input_size,fully_connected_num1],dtype=tf.float32,name='full1_weight')
        full1_bias = bias_variable(shape=[fully_connected_num1],dtype=tf.float32,name='full1_bias')
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(input_x,full1_weight),full1_bias))
    #全连接2
    with tf.variable_scope('full2') as scope:
        full2_weight = weight_variable(shape=[fully_connected_num1,fully_connected_num2],dtype=tf.float32,name='full2_weight')
        full2_bias = bias_variable(shape=[fully_connected_num2],name='full2_bias',dtype=tf.float32)
        full2_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1,full2_weight),full2_bias))
    # dropout
        # full2_layer2 = tf.nn.dropout(full2_layer2,keep_prob=keep_prob)
    # 全连接3
    with tf.variable_scope('full3') as scope:
        full3_weight = weight_variable(shape=[fully_connected_num2, 10],
                                           dtype=tf.float32, name='full3_weight')
        full3_bias = bias_variable(shape=[10], name='full3_bias', dtype=tf.float32)
        full3_layer3 = tf.nn.relu(tf.add(tf.matmul(full2_layer2, full3_weight), full3_bias))

    return full3_layer3


# 训练
def cnn_cifar10(x,labels,test,label_test):

    #占位
    input_shape=[None,image_width,image_height,num_channels]

    xPlace = tf.placeholder(shape=input_shape,dtype=tf.float32,name='xPlace')
    labelPlace = tf.placeholder(shape=[None],dtype=tf.int64,name='lablePlace')

    pred = cnn_model(xPlace)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,labels=labelPlace))

    #动态学习率改变
    # generation_num = tf.Variable(0, trainable=False)
    # model_learning_rate = tf.train.exponential_decay(learning_rate, generation_num, num_gens_to_wait, lr_decay,
    #                                                  staircase=True)
    myOpt = tf.train.AdamOptimizer(learning_rate)
    trainProcess = myOpt.minimize(loss,name='trainProcess')

    accur = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),labelPlace),tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(train_step):
            rand_index = np.random.choice(len(x),batch_size,replace=False)
            randx = x[rand_index,:,:,:]
            randLabel = labels[rand_index]
            # randx = cope_img(randx)
            # print(randx)
            sess.run(trainProcess,feed_dict={xPlace:randx,labelPlace:randLabel})
            accurTest = []
            losses = []
            if i%100 ==0:
                for j in range(1):
                    rand_index_test = np.random.choice(len(test), 2000, replace=False)
                    rand_test = test[rand_index_test, :, :, :]
                    rand_label_test = label_test[rand_index_test]
                    accurTest.append(sess.run(accur,feed_dict={xPlace:rand_test,labelPlace:rand_label_test}))
                    losses.append(sess.run(loss,feed_dict={xPlace:rand_test,labelPlace:rand_label_test}))
                tmp = tf.reduce_mean(accurTest).eval()
                losse = tf.reduce_mean(losses).eval()
                print('%d : acc =  %0.5f ,loss = %0.4f'%(i,tmp,losse))

if __name__ == '__main__':
    filePath = r'C:\Users\Administrator\Desktop\deeplearning\code\SampleData\cifar-10-batches-py'

    dataSet, labelSet, dataSetTest, labelSetTest = loadData(filePath)
    # dataSet = cope_img(dataSet)
    labelSetTest = np.array(labelSetTest)
    cnn_cifar10(dataSet, labelSet, dataSetTest, labelSetTest)