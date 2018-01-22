import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def loadTxt(filePath):
    fileData = np.loadtxt(filePath)
    # print(fileData[:,:13].shape)
    return fileData

def predHousing(sample):

    trainNum = 506
    trainPercent = 0.7
    testPercent = 0.2
    trainStep = 1000
    learningRate = 1

    index = [i for i in range(506)]
    random.shuffle(index)
    trainIndex = index[:round(trainPercent*trainNum)]
    testIndex = index[round(trainNum*trainPercent):round((trainPercent+testPercent)*trainNum)]
    predIndex = index[round((trainPercent+testPercent)*trainNum):]


    x = np.square((sample[:,:13] - sample[:,:13].min(0))/(sample[:,:13].max(0) - sample[:,:13].min(0)))
    xSample = np.column_stack((x,np.ones(trainNum)))
    # xSample = np.column_stack((sample[:,:13],np.ones(trainNum)))

    # xSample = tf.nn.l2_normalize(xSample[:, :13], dim=0)

    trainSample = xSample[trainIndex]
    testSample = xSample[testIndex]
    predSample = xSample[predIndex]
    trainLabel = sample[:,13][trainIndex].reshape((len(trainIndex),1))
    testLabel = sample[:,13][testIndex].reshape((len(testIndex),1))
    predLabel = sample[:,13][predIndex].reshape((len(predIndex),1))
    # trainSample = np.column_stack((trainSample,np.ones(round(trainPercent*trainNum))))

    # print(trainSample.shape)
    xPlace = tf.placeholder(shape=[None,14],dtype=tf.float32,name='xPlace')
    labelPlace = tf.placeholder(shape=[None,1],dtype=tf.float32,name='labelPlace')

    W = tf.Variable(tf.random_normal(shape=[14,1]),dtype=tf.float32,name='W')

    pred = tf.matmul(xPlace, W)

    loss = tf.reduce_mean(tf.square(labelPlace - pred))

    myOpt = tf.train.AdamOptimizer(learningRate)

    trainProcess = myOpt.minimize(loss,name='trainProcess')

    losses = []
    losses1 = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(trainStep):
            sess.run(trainProcess,feed_dict={xPlace:trainSample,labelPlace:trainLabel})
            # sess.run(trainProcess,feed_dict={xPlace:testSample,labelPlace:testLabel})
            losses.append(sess.run(loss,feed_dict={xPlace:trainSample,labelPlace:trainLabel}))
            losses1.append(sess.run(loss,feed_dict={xPlace:predSample,labelPlace:predLabel}))
        w = sess.run(W,feed_dict={xPlace:trainSample,labelPlace:trainLabel})
        y = sess.run(pred,feed_dict={xPlace:predSample,labelPlace:predLabel})
        y = y.flatten()
        print(predLabel)
        linex = np.linspace(-3,6,1000)
        linex_y = np.linspace(0,50,51)
        plt.plot(linex_y,y,c= 'r',label = 'real')
        plt.plot(linex_y,predLabel,c ='g',label = 'pred')
        print(losses)
        plt.legend()
        plt.show()
        plt.plot(linex, losses,c = 'y',label = 'real')
        plt.plot(linex,losses1,c = 'r',label = 'pred')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    data = loadTxt('./data/housing')
    predHousing(data)