import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import pdb


# 超参
sampleNum = 100
trainStep = 1000
learningRate = 0.02
trainRate = 0.8

# 数据集
xSample = np.concatenate((np.random.normal(4,3,50),np.random.normal(-2,2,50)))
ySample = np.concatenate((np.random.normal(3,3,50),np.random.normal(-3,2,50)))
labelSample = np.concatenate((np.repeat(0,50),np.repeat(1,50)))

# 画出数据集
colors = np.array(['r','g'])
plt.scatter(xSample[labelSample == 1],ySample[labelSample == 1],c = colors[0])
plt.scatter(xSample[labelSample == 0],ySample[labelSample == 0],c = colors[1])

# 打乱下标
index = [i for i in range(sampleNum)]
random.shuffle(index)
trainIndex = index[:round(trainRate * sampleNum)]
testIndex = index[round(trainRate * sampleNum):]

# 获取训练集和测试集
trainXsample = xSample[trainIndex]
trainYsample = ySample[trainIndex]
trainLabel = labelSample[trainIndex]

testXsample = xSample[testIndex]
testYsample = ySample[testIndex]
testLabel = labelSample[testIndex]

# plt.show()

# 占位
xPlace = tf.placeholder(shape = [None],dtype = tf.float32,name = 'xPlace')
yPlace = tf.placeholder(shape = [None],dtype = tf.float32,name = 'yPlace')
labelPlace = tf.placeholder(shape = [None], dtype = tf.float32, name= 'labelPlace')


# 参数
W = tf.Variable(tf.random_normal(shape=[]),dtype = tf.float32,name='W')
b = tf.Variable(tf.random_normal(shape=[]),dtype = tf.float32,name='b')

pred = tf.add(tf.multiply(W, xPlace),b)
# pred  = yPlace - yPred

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labelPlace,logits=pred))

myOpt = tf.train.GradientDescentOptimizer(learningRate)

trainProcess = myOpt.minimize(loss,name='trianProcess')

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for i in range(trainStep):
        sess.run(trainProcess,feed_dict={xPlace:trainXsample,yPlace:trainYsample,labelPlace:trainLabel})
        sess.run(loss, feed_dict={xPlace:trainXsample,yPlace:trainYsample,labelPlace:trainLabel})
        sess.run(loss, feed_dict={xPlace:testXsample,yPlace:testYsample,labelPlace:testLabel})
    lastW = sess.run(W)
    lastb = sess.run(b)
    linex = np.linspace(-5,5,1000)
    # pdb.set_trace()
    plt.plot(linex,(linex * lastW + lastb))
    plt.show()