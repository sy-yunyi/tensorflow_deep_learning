from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
# import PIL as plt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import pdb


mnist = input_data.read_data_sets('G:/mnist',one_hot=True)

# print(mnist.train.images.shape)
# print(mnist.train.labels[:20,:])
# INPUT_NODE = 784
# OUTPUT_NODE = 10
#
# LAYER1_NODE =500
# BATCH_SIZE = 100
#
# learningRate = 0.8
# laerningRateDecay = 0.99
#
# RegularizationRate = 0.0001
# trainStep = 3000
#
# def inference(input_tensor,avg_class, weight1,biases1,weight2,biases2):
#     if avg_class == None:
#         layer1 = tf.nn.relu(tf.matmul(input_tensor,weight1)+biases1)
#         return tf.matmul(layer1,weight2)+biases2
#     else:
#         layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight1))+avg_class.average(biases1))
#         return tf.matmul(layer1,avg_class.average(weight2))+avg_class.average(biases2)
# def train(mnist):
#     x = tf.placeholder(shape=[None,INPUT_NODE],dtype=tf.float32,name='x')
#     y_ = tf.placeholder(shape=[None,OUTPUT_NODE],dtype=tf.float32,name='y')
#
#     weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
#     biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
#
#     weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev = 0.1))
#     biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))


def loadImg():
    im = Image.open('./data/1.png')
    im = im.resize((28,28)).convert('L')
    # im.show()
    # print(im.format, im.size, im.mode)
    img = np.array(im)
    # print(img)
    # img1 = [1. if i == False else 0 for item in img for i in item]
    img = 1.0 - img / 255.
    # print(img)
    # print(np.array(img1).reshape(1,-1).shape)
    # print(np.cast(img))
    # norm_img = (img-img.min(0))/(img.max(0) - img.min(0))
    # print(norm_img)
    return np.array(img).reshape(1,-1)


def mnistClassifiy(mnist):

    trainStep = 50
    learningRate = 1
    trainNum = mnist.train.num_examples
    batch_size = 11000

    xPlace = tf.placeholder(shape=[None,784],dtype=tf.float32,name='xPlace')
    labelPlace = tf.placeholder(shape=[None,10],dtype=tf.float32,name='labelPlace')

    W = tf.Variable(tf.random_normal(shape=[784,10]),dtype=tf.float32,name='W')
    # W1 = tf.Variable(tf.random_normal(shape=[784,10]),dtype=tf.float32,name='W1')

    b = tf.Variable(tf.random_normal(shape=[]),dtype=tf.float32,name='b')

    pred =tf.add(tf.matmul(xPlace,W),b)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labelPlace,logits=pred)

    vec = tf.nn.softmax(logits=pred)

    accur = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits=pred),axis =1),tf.argmax(labelPlace,axis=1)),tf.float32))

    my_opt = tf.train.AdamOptimizer(learningRate)
    trainProcess = my_opt.minimize(loss,name='trainProcess')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(trainStep):
            batch_num = int(trainNum/batch_size)
            # print(batch_num)
            for j in range(batch_num):
                batch_xs,batch_ys = mnist.train.next_batch(batch_size)
                sess.run(trainProcess,feed_dict={xPlace:batch_xs,labelPlace:batch_ys})
                # sess.run(trainProcess,feed_dict={xPlace:mnist.test.images,labelPlace:mnist.test.labels})
            accues = sess.run(accur, feed_dict={xPlace: mnist.test.images, labelPlace: mnist.test.labels})
            print('第 %d 次循环的正确率是：%0.6f%%'%(i+1,accues*100))
        # vect = sess.run(vec,feed_dict={xPlace:mnist.test.images,labelPlace:mnist.test.labels})
        saver = tf.train.Saver()
        saver_path = saver.save(sess, './model/model')
        lastb = sess.run(b)
        print(lastb)
        # print(vect)


def mnistClassifiy2(mnist,img =None):
    trainStep = 200
    learningRate = 0.0002
    trainNum = mnist.train.num_examples
    hideNode = 200
    xPlace = tf.placeholder(shape=[None, 784], dtype=tf.float32, name='xPlace')
    labelPlace = tf.placeholder(shape=[None], dtype=tf.int64, name='labelPlace')

    W = tf.Variable(tf.random_normal(shape=[784, hideNode]), dtype=tf.float32, name='W')

    W1 = tf.Variable(tf.random_normal(shape=[hideNode,10]),dtype=tf.float32,name='W1')

    b = tf.Variable(tf.random_normal(shape=[hideNode]), dtype=tf.float32, name='b')
    b2 = tf.Variable(tf.random_normal(shape=[1]), dtype=tf.float32, name='b2')

    hidePred =tf.nn.tanh(tf.add(tf.matmul(xPlace, W), b))
    pred = tf.add(tf.matmul(hidePred,W1),b2)


    # print(pred.shape)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labelPlace, logits=pred)

    vec = tf.nn.softmax(logits=pred)

    accur = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits=pred), axis=1), labelPlace), tf.float32))

    my_opt = tf.train.GradientDescentOptimizer(learningRate)
    trainProcess = my_opt.minimize(loss, name='trainProcess')

    losses = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(trainStep):
            sess.run(trainProcess, feed_dict={xPlace: mnist.train.images, labelPlace: mnist.train.labels})
            sess.run(trainProcess, feed_dict={xPlace: mnist.test.images, labelPlace: mnist.test.labels})
            if (i%100 ==0):
                accues = sess.run(accur, feed_dict={xPlace: mnist.test.images, labelPlace: mnist.test.labels})
                print('第 %d 次循环的正确率是：%0.6f%%' % (i + 1, accues * 100))
        saver = tf.train.Saver()
        saver_path = saver.save(sess,'./model/model')
        lastW, lastb2 = sess.run([W,b2])
        pred1 = sess.run(pred,feed_dict={xPlace:img})
        print(lastb2)
        print(pred1)
        pdb.set_trace()
        print(tf.argmax(pred1,1).eval())

def mnistPred(img):
    sess = tf.Session()
    graph = tf.get_default_graph()

    saver = tf.train.import_meta_graph('./model/model.meta')
    # xPlace = tf.placeholder(shape=[None, 784], dtype=tf.float32, name='xPlace')
    # labelPlace = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='labelPlace')
    #
    # W = tf.Variable(tf.random_normal(shape=[784, 10]), dtype=tf.float32, name='W')
    # b = tf.Variable(tf.random_normal(shape=[]), dtype=tf.float32, name='b')
    xPlace = tf.convert_to_tensor(img,dtype=tf.float32)
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess,tf.train.latest_checkpoint('model/'))

    A = graph.get_tensor_by_name('W:0')
    B = graph.get_tensor_by_name('b:0')
    pred =tf.add(tf.matmul(xPlace, A), B)

    # print(sess.run(b))
    # pred1 = sess.run(pred, feed_dict={xPlace: img})
    print(pred.eval(session=sess)[0])
    print('预测值为：', np.argmax(pred.eval(session=sess)[0]))

def predClassifiy(img):
    hideNode = 100
    graph = tf.get_default_graph()
    # W = tf.Variable(tf.random_normal(shape=[784, 10]), dtype=tf.float32, name='W')
    # b = tf.Variable(tf.random_normal(shape=[]), dtype=tf.float32, name='b')
    # saver = tf.train.Saver()
    saver = tf.train.import_meta_graph('./model/model.meta')
    xPlace = tf.placeholder(shape=[None, 784], dtype=tf.float32, name='xPlace')
    W = tf.Variable(tf.random_normal(shape=[784, hideNode]), dtype=tf.float32, name='W')

    W1 = tf.Variable(tf.random_normal(shape=[hideNode, 10]), dtype=tf.float32, name='W1')

    b = tf.Variable(tf.random_normal(shape=[hideNode]), dtype=tf.float32, name='b')
    b2 = tf.Variable(tf.random_normal(shape=[1]), dtype=tf.float32, name='b2')

    hidePred = tf.nn.tanh(tf.add(tf.matmul(xPlace, W), b))
    pred = tf.add(tf.matmul(hidePred, W1), b2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,tf.train.latest_checkpoint('model/'))

        W = W.assign(graph.get_tensor_by_name('W:0'))
        b = b.assign(graph.get_tensor_by_name('b:0'))
        # pdb.set_trace()
        W1 = W1.assign(graph.get_tensor_by_name('W1:0'))
        b2 = b2.assign(graph.get_tensor_by_name('b2:0'))

        # W1 = sess.run(W)
        b1 = sess.run(b2)
        print(b1)
        pred1 = sess.run(pred,feed_dict={xPlace:img})
        # pdb.set_trace()
        print('预测值为：',tf.argmax(pred1,1).eval()[0])





if __name__ == '__main__':
    img = loadImg()
    # 训练 one_hot
    mnistClassifiy(mnist)

    #进行训练，非one_hot
    # mnistClassifiy2(mnist,img)
    # 预测
    # predClassifiy(img)
    # mnistPred(img)

