import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
import pdb
def loadData(filePath):
    """
    读取数据
    :param fileName: 文件路径
    :return:
    """
    data_file = pd.read_csv(filePath,header=None)
    xSample = data_file[2].values
    ySample = data_file[3].values
    zSample = data_file[1].values
    # lableSample = pd.get_dummies(data_file[4]).values
    # lableSample =[0 if data =='Iris-setosa' else 1 if data == 'Iris-versicolor'  else 2 for data in  data_file[4].values]
    lableSample =[0. if data =='Iris-setosa' else 1. for data in  data_file[4].values]

    return xSample,ySample,zSample,lableSample

def backPropagation(xSample,ySample):
    """
    两个特征的拟合
    :param xSample: 特征X
    :param ySample: 特征Y
    :return: None
    """

    # 超参
    trainNum = 150
    trainPercent = 0.8
    learningRate = 0.001
    trainStep = 1000

    # 画出数据点
    plt.scatter(xSample,ySample)

    # 获取随机的下标
    index = [i for i in range(trainNum)]
    random.shuffle(index)
    trainIndex = index[:round(trainPercent * trainNum)]
    testIndex = index[round(trainPercent* trainNum):]

    # 随机取得训练集和测试集
    trainXsample = xSample[trainIndex]
    trainYsample = ySample[trainIndex]
    testXsample = xSample[testIndex]
    testYsample = ySample[testIndex]

    # 占位
    xPlace = tf.placeholder(shape=[None],dtype=tf.float32,name='xPlace')
    yPlace = tf.placeholder(shape=[None],dtype=tf.float32,name='yPlace')

    # 参数
    W = tf.Variable(tf.random_normal(shape=[]),dtype=tf.float32,name='W')
    b = tf.Variable(tf.random_normal(shape=[]),dtype=tf.float32,name='b')
    # 函数模型
    pred = tf.add(tf.multiply(W,xPlace),b)
    # loss
    loss = tf.reduce_mean(tf.square(yPlace - pred))
    #选择优化器，梯度下降
    myOpt= tf.train.AdamOptimizer(learningRate)
    trainProcess = myOpt.minimize(loss,name= 'trainProcess')

    lossesTrain =[]
    lossesTest = []
    #进行训练
    with tf.Session() as sess :
        #初始化全部变量
        sess.run(tf.global_variables_initializer())
        for i in range(trainStep):
            # print(sess.run(W))

            sess.run(trainProcess,feed_dict={xPlace:trainXsample,yPlace:trainYsample})
            lossesTrain.append(sess.run(loss,feed_dict={xPlace:trainXsample,yPlace:trainYsample}))
            lossesTest.append(sess.run(loss,feed_dict={xPlace:testXsample,yPlace:testYsample}))


        # 获取最后的参数值
        lastW = sess.run(W)
        lastb = sess.run(b)

        #画出拟合曲线
        linex = np.linspace(0,8,1000)
        plt.plot(linex,linex * lastW + lastb)
        plt.show()
        plt.plot(np.linspace(-1,10,1000),lossesTrain,c = 'g',label = 'train')
        plt.plot(np.linspace(-1,10,1000),lossesTest,c = 'r' ,label = 'test')
        plt.legend()
        plt.show()

def Decomposition(xSample,ySample):
    """
    最小二乘法求解
    [W,b]=(X.T *X).I * X.T * Y    (矩阵运算)
    :param xSample: 特征X
    :param ySample: 特征Y
    :return: None
    """
    # 超参
    trainNum = 150
    trainPercent = 0.8
    learningRate = 0.001
    trainStep = 1000
    plt.scatter(xSample, ySample)
    # 处理数据，使得可以进行矩阵运算
    X = np.column_stack((xSample.reshape(-1, 1), np.ones(150).reshape(-1, 1)))
    Y = ySample.reshape(-1, 1)

    #最小二乘法计算
    mulMat = tf.matmul(tf.transpose(X),X)
    inverMat = tf.matrix_inverse(mulMat)
    mulMatAndI = tf.matmul(inverMat,tf.transpose(X))
    solustion = tf.matmul(mulMatAndI,Y)

    with tf.Session() as sess :
        [W,b] = sess.run(solustion)
        linex = np.linspace(0,8,1000)
        plt.plot(linex,linex * W + b)
        plt.show()

def draw3D(xSample,ySample,zSample,w1,w2,b,lables=None):

    ax = plt.subplot(111,projection = '3d')
    ax.scatter(xSample,ySample,zSample)

    x = np.linspace(-8,8,num=100)
    y = np.linspace(-8,8,num=100)

    x, y = np.meshgrid(x, y)

    z = w1 * x + w2 * y +b
    ax.plot_surface(x,y,z)
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    plt.show()


def backPropagationThree(xSample,ySample,zSample):
    # 超参
    trainNum = 150
    trainPercent = 0.8
    learningRate = 0.0002
    trainStep = 1000

    # 获取随机的下标
    index = [i for i in range(trainNum)]
    random.shuffle(index)
    trainIndex = index[:round(trainPercent * trainNum)]
    testIndex = index[round(trainPercent * trainNum):]

    # 随机取得训练集和测试集
    trainXsample = xSample[trainIndex]
    trainYsample = ySample[trainIndex]
    trainZsample = zSample[trainIndex]
    testZsample = zSample[testIndex]
    testXsample = xSample[testIndex]
    testYsample = ySample[testIndex]


    xPlace = tf.placeholder(shape=[None],dtype=tf.float32,name='xPlace')
    yPlace = tf.placeholder(shape = [None],dtype=tf.float32,name='yPlace')
    zPlace = tf.placeholder(shape=[None],dtype=tf.float32,name='zPlace')


    W1 = tf.Variable(tf.random_normal(shape=[]),dtype = tf.float32,name='W1')
    W2 = tf.Variable(tf.random_normal(shape=[]),dtype=tf.float32,name='W2')
    b = tf.Variable(tf.random_normal(shape=[]),dtype= tf.float32,name='b')

    pred = tf.add(tf.add(tf.multiply(W1,xPlace),tf.multiply(W2,yPlace)),b)

    loss = tf.reduce_mean(tf.square(pred - zPlace))

    myOpt = tf.train.AdamOptimizer(learningRate)
    trainProcess = myOpt.minimize(loss,name='trainProcess')

    losses = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(trainStep):
            sess.run(trainProcess,feed_dict={xPlace:xSample,yPlace:ySample,zPlace:zSample})
            losses.append(sess.run(loss,feed_dict={xPlace:xSample,yPlace:ySample,zPlace:zSample}))
        lastW1 = sess.run(W1)
        lastW2 = sess.run(W2)
        lastb = sess.run(b)
        print(lastW1,lastW2,lastb)
        draw3D(xSample,ySample,zSample,lastW1,lastW2,lastb)
        linex = np.linspace(-4,4,1000)
        plt.plot(linex,losses)
        plt.show()


def classifiy(xSample, ySample, lableSample):

    #超参
    trainNum = 150
    trainStep = 1000
    trainPercent = 0.8
    learningRate = 0.001

    plt.scatter(xSample[lableSample[:,0] == 1],ySample[lableSample[:,0] == 1],c = 'r')
    plt.scatter(xSample[lableSample[:,1] == 1],ySample[lableSample[:,1] == 1],c = 'y')
    plt.scatter(xSample[lableSample[:,2] == 1],ySample[lableSample[:,2] == 1],c = 'b')
    # plt.show()

    X = np.column_stack((xSample,ySample))

    xPlace = tf.placeholder(shape=[None,2],dtype=tf.float32,name='xPlace')
    # yPlace = tf.placeholder(shape=[None],dtype=tf.float32,name='yPlace')
    lPlace = tf.placeholder(shape=[None,3],dtype=tf.float32,name='lPlace')


    W = tf.Variable(tf.random_normal(shape=[2,3]),dtype=tf.float32,name='W')
    b = tf.Variable(tf.random_normal([3]),dtype = tf.float32,name='b')

    pred = tf.nn.bias_add(tf.matmul(xPlace,W),b)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=lPlace))

    myOpt = tf.train.GradientDescentOptimizer(learningRate)
    trainProcess = myOpt.minimize(loss,name='trainProcess')

    losses = []
    with tf.Session() as sess :
        sess.run(tf.global_variables_initializer())
        for i in range(trainNum):
            sess.run(trainProcess,feed_dict={xPlace:X,lPlace:lableSample})
        lastW = sess.run(W)
        lastb = sess.run(b)
        print(lastW)
        print(lastb)

        linex = np.linspace(0,5,1000)
        plt.plot(linex,-(linex * lastW[0][0] + lastb[0])/lastW[1][0])
        plt.plot(linex,-(linex * lastW[0][1]+ lastb[1])/lastW[1][1])
        # plt.plot(linex,-(linex * lastW[0][2]+ lastb[2])/lastW[1][2])
        plt.show()

def classifiyThree(xSample,ySample,zSample,lableSample):

    #超参
    trainStep = 1000
    learningRate = 0.001
    trainPercent = 0.8
    trainNum = 150
    ax = plt.subplot(111, projection='3d')
    lableSample = np.array(lableSample)
    ax.scatter(xSample[lableSample == 1], ySample[lableSample == 1], zSample[lableSample == 1],c='r',marker='*',label='iris')
    ax.scatter(xSample[lableSample == 0], ySample[lableSample == 0], zSample[lableSample == 0],c='g',marker='>',label='iris_two')
    lableSample = np.array(lableSample).reshape((trainNum, 1))

    X = np.column_stack((np.column_stack((np.column_stack((xSample,ySample)),zSample)),np.zeros(trainNum)))

    xPlace = tf.placeholder(shape=[None,4],dtype=tf.float32,name='aPlace')
    lablePlace = tf.placeholder(shape=[None,1],dtype=tf.float32,name='lablePlace')

    W1 = tf.Variable(tf.random_normal(shape=[4,1]),dtype=tf.float32,name='W1')

    y = tf.matmul(xPlace,W1)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=lablePlace,logits=y)

    myOpt = tf.train.GradientDescentOptimizer(learningRate)
    trainProcess = myOpt.minimize(loss,name='trainProcess')
    tf.nn.batch_normalization()

    losses = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(trainStep):
            sess.run(trainProcess,feed_dict={xPlace:X,lablePlace:lableSample})
            losses.append(sess.run(loss,feed_dict={xPlace:X,lablePlace:lableSample}))

        w = sess.run(W1)
        linex = np.linspace(-4,8,1000)
        y = (w[0] * linex +w[3]) /(-w[1])
        x, y = np.meshgrid(linex, y)
        z = (w[0] * x +w[1] * y +w[3])/(-w[2])
        print(z)
        ax.plot_surface(x,y,z)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    filePath = './data/iris'
    x, y, z, lables =loadData(filePath)
    # 两特征拟合
    # backPropagation(x,y)
    # 最小二乘法拟合
    # Decomposition(x,y)
    # 绘制3维图
    # draw3D(x,y,z)
    # 三特征拟合
    # backPropagationThree(x, y, z)
    #平面分三类
    # classifiy(x,y,lables)
    classifiyThree(x,y,z,lables)


