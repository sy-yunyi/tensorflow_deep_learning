import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# 归一化
# (z-min)/(max-min)
def normalization(z):
    return (z - z.min(axis=0)) / (z.max(axis=0) - z.min(axis=0))

# create dataset tp1, tp2
tp1 = np.column_stack((np.random.normal(1., 1, 100),
                 np.random.normal(1., 1, 100), np.ones(shape=[100])))

tp2 = np.column_stack((np.random.normal(5., 1, 100),
                 np.random.normal(5., 1, 100), np.zeros(shape=[100])))

tp3 = np.column_stack((np.random.normal(4., 1, 100),
                 np.random.normal(8., 1, 100), 2*np.ones(shape=[100])))


plt.plot(tp1[:, 0], tp1[:, 1], 'r+')
plt.plot(tp2[:, 0], tp2[:, 1], 'bo')
plt.plot(tp3[:, 0], tp3[:, 1], 'g*')

# plt.show()

# 超参
sampleNum = tp1.shape[0] + tp1.shape[0] +tp3.shape[0]
trainStep = 500
learningRate = 2
trainPercent = 1
testPercent = 0

# 样本数据, 归一化， 加一列1
xSample = np.concatenate((tp1[:, :-1], tp2[:, :-1],tp3[:, :-1]))
labelSample = np.concatenate((tp1[:, -1], tp2[:, -1],tp3[:,-1])).reshape([-1, 1])

# 获取train、test、 pred 乱序index
index = [i for i in range(sampleNum)]
random.shuffle(index)
trainIndex = index[:round(trainPercent*sampleNum)]
testIndex = index[round(trainPercent*sampleNum):]

# train set、 test set、 pred set
trainXsample = xSample[trainIndex]
trainLabelsample = labelSample[trainIndex]

testXsample = xSample[testIndex]
testLabelsample = labelSample[testIndex]

# define placeholder
xPlace = tf.placeholder(shape=[None, 2], dtype=tf.float32)
labelPlace = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 参数
W = tf.Variable(tf.random_normal(shape=[2, 1]))
b = tf.Variable(tf.random_normal(shape=[]))

# 假设公式, 注意是矩阵运算
pred = tf.add(tf.matmul(xPlace, W), b)

# define loss
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,
                                                              labels=labelPlace))
# 训练过程
trainProcess = tf.train.AdamOptimizer(learningRate).\
    minimize(tf.reduce_mean(loss))

# 准确率
accur = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.nn.sigmoid(pred)), labelPlace), tf.float32))

# 开始训练
lossesTrain = []
lossesTest = []
accurs = []
for i in range(3):
    trainLabelsample = np.array([1. if item == i else 0 for item in trainLabelsample]).reshape(len(trainLabelsample),1)
    with tf.Session() as sess:
        # 初始化 Variables
        sess.run(tf.global_variables_initializer())
        for step in range(trainStep):
            sess.run(trainProcess, feed_dict={
                xPlace: trainXsample,
                labelPlace: trainLabelsample})
            lossesTrain.append(sess.run(loss, feed_dict={
                xPlace: trainXsample,
                labelPlace: trainLabelsample}))
        lastW, lastb  = sess.run([W,b])
        x = np.linspace(start = np.min(xSample),stop=np.max(xSample),num = 100)
        plt.plot(x,(lastW[0][0]*x+lastb)/(-lastW[1][0]),label = i)
# plt.plot(range(trainStep), lossesTrain, '-', label='lossTrain')
# plt.plot(range(trainStep), lossesTest, '-', label='lossTest')
# plt.plot(range(trainStep), accurs, '-', label='accur')
plt.legend()
plt.show()
