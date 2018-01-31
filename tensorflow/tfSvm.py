import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def loadData(filePath):
    """
    读取数据
    :param fileName: 文件路径
    :return:
    """
    data_file = pd.read_csv(filePath,header=None)
    x = data_file.values[:,:4]
    labelSample =[[1. if data =='Iris-setosa' else -1. for data in  data_file[4].values]]
    labelSample = np.array(labelSample)

    return x,labelSample

def svn_tf (x,labels):

    delta = 1.
    x = np.transpose(x)
    xG  = []
    for i in range(x.shape[1]):
        X = x.T
        X = X.astype(float)
        g = np.exp(-1.0 * np.sum((X[i]-X)**2,axis=1)/3)
        xG.append(g)
    x = np.array(xG).T
    lx = x.shape[0]
    xPlace = tf.placeholder(shape=[lx,None],dtype=tf.float32,name='xPlace')
    yPlace = tf.placeholder(shape=[1,None],dtype=tf.float32,name='yPlace')


    W = tf.Variable(tf.random_normal(shape=[1,lx]),dtype=tf.float32,name='W')
    b = tf.Variable(tf.random_normal(shape=[]),dtype=tf.float32,name='b')



    pred =  tf.matmul(W,xPlace)+b

    lossW = tf.reduce_mean(tf.square(W))
    lossLabel = tf.reduce_mean(tf.maximum(0.,1.-yPlace * (tf.matmul(W,xPlace)+b)))

    loss = lossW+lossLabel

    myopt = tf.train.GradientDescentOptimizer(0.01)
    trainprocess = myopt.minimize(loss,name='trainProcess')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            sess.run(trainprocess,feed_dict={xPlace:x,yPlace:labels})
        lastW, lastb = sess.run([W,b])
        p = sess.run(pred,feed_dict={xPlace:x,yPlace:labels})

    y_predict = np.dot(lastW, x) + lastb
    print('Accuracy of SVM in Tensorflow is:{}'.format(np.sum((y_predict > 0) == (labels > 0)) * 100 / len(labels[0])))


if __name__ == '__main__':
    filePath = './data/iris'
    x, labelSample = loadData(filePath)
    svn_tf(x,labelSample)