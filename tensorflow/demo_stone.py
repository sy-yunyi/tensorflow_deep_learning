import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tensorflow as tf
import random
import pandas as pd
def loadData():
    filePath = 'E:\data'
    labels =[]
    sample = []
    for root,dir,file in os.walk(r'E:\data'):
        for name in file:
            labels.append(int(name[0]))
            img = Image.open(filePath+'/'+name)
            # print(img.size)
            img_data = np.array(img.convert('L').resize((50,50))).reshape(1,-1)
            sample.append(img_data[0])
    return sample,labels


def load():
    img = Image.open('E:/test/1 (2).jpg')
    return np.array(img.convert('L').resize((50,50))).reshape(1,-1)[0]


def classifiyStone(sample,label,img= None):

    trainStep = 150
    learningRate = 1
    trainNum = len(sample)

    trainPrence = 0.9
    testPrecen = 0.1

    sample = np.array(sample)
    labelSample = np.array(label)
    labelSample = pd.get_dummies(labelSample).values

    index = [i for i in range(trainNum)]
    random.shuffle(index)

    trainIndex = index[:round(trainNum * trainPrence)]
    testIndex = index[round(trainNum * trainPrence):round(trainNum * (trainPrence + testPrecen))]
    trainSample = sample[trainIndex]
    trainLabel = labelSample[trainIndex,:]
    testSample = sample[testIndex]
    testLable = labelSample[testIndex,:]


    # 占位
    xPlace = tf.placeholder(shape=[None, 2500], dtype=tf.float32, name='xPlace')
    labelPlace = tf.placeholder(shape=[None, 3], dtype=tf.float32, name='labelPlace')

    # 变量
    W = tf.Variable(tf.random_normal(shape=[2500, 3]), dtype=tf.float32, name='W')

    b = tf.Variable(tf.random_normal(shape=[]), dtype=tf.float32, name='b')

    # 预测模型
    pred = tf.add(tf.matmul(xPlace, W), b,name='pred')
    # 损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labelPlace, logits=pred))
    # 准确率
    accur = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits=pred), axis=1), tf.argmax(labelPlace, axis=1)), tf.float32))
    # 优化器
    my_opt = tf.train.AdamOptimizer(learningRate)
    trainProcess = my_opt.minimize(loss, name='trainProcess')

    losses_tr = []
    losses_te = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #训练
        for i in range(trainStep):
            sess.run(trainProcess,feed_dict={xPlace:trainSample,labelPlace:trainLabel})
            losses_tr.append(sess.run(loss, feed_dict={xPlace: trainSample, labelPlace: trainLabel}))
            losses_te.append(sess.run(loss,feed_dict={xPlace:testSample,labelPlace:testLable}))

        accues = sess.run(accur,feed_dict={xPlace:testSample,labelPlace:testLable})

        print('准确率为：%0.5f%% ' % (accues*100))
        cl = sess.run(pred,feed_dict={xPlace:img})
        if np.argmax(cl[0]) == 0:
            print('预测结果是：包袱' )
        elif np.argmax(cl[0]) == 1:
            print('预测结果是：石头' )
        elif np.argmax(cl[0]) == 2:
            print('预测结果是：剪刀')
        saver = tf.train.Saver()
        saver.save(sess,'stone/test')
        linex = np.linspace(-3,5,trainStep)
        plt.plot(linex,losses_tr,'r-',label = 'train')
        plt.plot(linex, losses_te, 'g--', label='test')
        plt.legend()
        plt.show()


def stoneDamo(img):
    sess = tf.Session()
    saver = tf.train.import_meta_graph('stone/test.meta')
    saver.restore(sess,'stone/test')
    graph = tf.get_default_graph()
    xPlace = graph.get_tensor_by_name('xPlace:0')
    labelPlace = graph.get_tensor_by_name('labelPlace:0')
    pred = graph.get_tensor_by_name('pred:0')

    p = sess.run(pred,feed_dict={xPlace:img})
    if np.argmax(p[0]) == 0:
        print('预测结果是：包袱')
    elif np.argmax(p[0]) == 1:
        print('预测结果是：石头')
    elif np.argmax(p[0]) == 2:
        print('预测结果是：剪刀')


if __name__ == '__main__':
    sample, label = loadData()
    img = load()
    # print(img.reshape(1,-1))
    # classifiyStone(sample,label,img.reshape(1,-1))
    stoneDamo(img.reshape((1,2500)))