import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def normalization(z):
    return ((z - z.min(0))/(z.max(0)-z.min(0)))

data = np.loadtxt('./data/training')
# print(type(data))
xTrain = normalization(data[:12,0]).reshape((12,1))
yTrain = normalization(data[:12,1]).reshape((12,1))

xVal = normalization(data[12:33,0]).reshape((21,1))
yVal = normalization(data[12:33,1]).reshape((21,1))

# plt.plot(xTrain,yTrain,'+')
# plt.plot(xVal,yVal,'*')
# plt.show()

def train(xTrain,yTrain,xVal,yVal,n=1):
    """

    :param xTrain:
    :param yTrain:
    :param xVal:
    :param yVal:
    :param n:
    :return:
    """

    xPlace = tf.placeholder(shape=[None,1],dtype = tf.float32,name='xPlace')
    yPlace = tf.placeholder(shape=[None,1],dtype=tf.float32,name='yPlace')

    W = tf.Variable(tf.ones(shape=[n,1]),dtype=tf.float32,name='W')
    b = tf.Variable(tf.constant(1.),dtype=tf.float32,name='b')

    tmpxp = xPlace
    for i in range(2,n+1):
        tmpxp = tf.concat([tmpxp,tf.pow(xPlace,i)],1)

    pred = tf.add(tf.matmul(tmpxp,W),b,name='pred')

    loss = tf.reduce_mean(tf.square(pred - yPlace))

    myopt = tf.train.AdamOptimizer(1)
    trainPrecoss = myopt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(2000):
            sess.run(trainPrecoss,feed_dict={xPlace:xTrain,yPlace:yTrain})
        trainLoss = sess.run(loss,feed_dict={xPlace:xTrain,yPlace:yTrain})
        valLoss = sess.run(loss,feed_dict={xPlace:xVal,yPlace:yVal})

        lastW = sess.run(W)
        lastb = sess.run(b)
    return trainLoss,valLoss,lastW,lastb



def train_test():
    trainLoss = []
    valLoss = []

    for i in range(2, 4):
        currTrainLoss, currValLoss, lastW, lastb = train(xTrain, yTrain, xVal, yVal, i)
        trainLoss.append(currTrainLoss)
        valLoss.append(currValLoss)

    print(lastW.shape, lastb)
    linex = np.linspace(0, 1, 50)
    y1 = linex * lastW[0]
    y2 = linex * linex * lastW[1]
    y3 = linex * linex * linex * lastW[2]
    y = y1 + y2 + y3 + lastb
    plt.plot(xTrain, yTrain, '+')
    plt.plot(xVal, yVal, '*')
    # y = linex.reshape(-1,1)*lastW +lastb
    # print(y)
    plt.plot(linex, y)

    # linex = np.linspace(0,10,len(trainLoss))
    # plt.plot(linex,trainLoss)
    # plt.plot(linex,valLoss)
    plt.show()

def train_sample(xTrain,yTrain,xVal,yVal,n=1):

    xPlace = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='xPlace')
    yPlace = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='yPlace')

    W = tf.Variable(tf.ones(shape=[2, 1]), dtype=tf.float32, name='W')
    b = tf.Variable(tf.constant(1.), dtype=tf.float32, name='b')

    tmpxp = xPlace
    for i in range(2, n+1):
        tmpxp = tf.concat([tmpxp, tf.pow(xPlace, i)], 1)

    pred = tf.add(tf.matmul(tmpxp, W), b, name='pred')

    loss = tf.reduce_mean(tf.square(pred - yPlace))+tf.contrib.layers.apply_regularization()

    myopt = tf.train.AdamOptimizer(1)
    trainPrecoss = myopt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(100):
            sess.run(trainPrecoss, feed_dict={xPlace: xTrain, yPlace: yTrain})
        trainLoss = sess.run(loss, feed_dict={xPlace: xTrain, yPlace: yTrain})
        valLoss = sess.run(loss, feed_dict={xPlace: xVal, yPlace: yVal})

        lastW = sess.run(W)
        lastb = sess.run(b)
    return trainLoss, valLoss, lastW, lastb


def test_sample():
    trainLoss = []
    valLoss = []

    for i in range(4, 13):
        currTrainLoss, currValLoss, lastW, lastb = train(xTrain[:i,:], yTrain[:i,:], xVal[:i,:], yVal[:i,:],n=50)
        trainLoss.append(currTrainLoss)
        valLoss.append(currValLoss)

    linex = np.linspace(4,12,len(trainLoss))
    plt.plot(linex,trainLoss)
    plt.plot(linex,valLoss)
    plt.show()


if __name__ == '__main__':
    # train_test()
    test_sample()



