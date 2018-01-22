import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

def loadTxt(filePath):
    fileData = np.loadtxt(filePath)[:,1:]
    return fileData[:,:-1],fileData[:,-1]

def lowbwt(sample,label):
    # 超参
    trainPrecent = 0.5
    trainStep = 5000
    trainNum = len(sample)
    learningRate = 4

    sample = (sample - sample.min(0))/(sample.max(0) - sample.min(0))
    # sample = np.column_stack((sample,np.ones(trainNum)))
    # print(sample)
    index = [i for i in range(trainNum)]
    random.shuffle(index)

    trainIndex = index[:round(trainNum * trainPrecent)]
    testIndex = index[round(trainPrecent * trainNum):]
    trainSample = sample[trainIndex]
    trainLabel = label[trainIndex].reshape((len(trainIndex),1))
    testSample = sample[testIndex]
    testLabel = label[testIndex].reshape((len(testIndex),1))
    # print(trainLabel)
    xPlace = tf.placeholder(shape=[None,9],dtype = tf.float32,name='xPlace')
    labelPlace = tf.placeholder(shape=[None,1],dtype=tf.float32,name='labelPlace')

    W = tf.Variable(tf.random_normal(shape=[9,1]),dtype=tf.float32,name='W')
    b = tf.Variable(tf.random_normal(shape=[]),dtype=tf.float32,name='b')

    pred = tf.add(tf.matmul(xPlace,W),b)
    # pred = tf.matmul(xPlace,W)
    loss = tf.reduce_mean(tf.square(pred - labelPlace))

    myOpt = tf.train.AdamOptimizer(learningRate)
    trainProcess = myOpt.minimize(loss ,name='trainProcess')

    losses = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(trainStep):
            sess.run(trainProcess,feed_dict={xPlace:trainSample,labelPlace:trainLabel})
            losses.append(sess.run(loss,feed_dict={xPlace:trainSample,labelPlace:trainLabel}))

        predy = sess.run(pred,feed_dict={xPlace:testSample})
        linex = np.linspace(-1,9,len(testSample))
        print(losses)
        print(np.mean(np.abs(predy - testLabel)))
        plt.plot(linex,testLabel,c = 'r',label='real')
        plt.plot(linex,predy,c= 'g',label='pred')
        plt.legend()
        plt.show()
        plt.plot(np.linspace(-1,9,trainStep),losses)
        plt.show()

if __name__ == '__main__':
    filePath = './data/lowbwt'
    sample,label = loadTxt(filePath)
    lowbwt(sample, label)
