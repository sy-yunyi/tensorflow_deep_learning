import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
import random
from urllib import request
from PIL import ImageEnhance
import PIL
from PIL import Image
def loadData(filePath):
    """
    读取数据
    :param fileName: 文件路径
    :return:
    """
    data_file = pd.read_csv(filePath,header=None)
    xSample = data_file[1].values
    ySample = data_file[3].values
    # plt.scatter(xSample,ySample)
    # plt.show()
    sample = np.column_stack((xSample,ySample))
    # print(sample.shape)
    return sample

def tfKmeans(sample,n):
    """
    使用tensorflow实现Kmeans算法
    :param sample: 数据集
    :param n: 分类数
    :return: None
    """
    # 随机选取初始聚类点
    index = list(range(len(sample)))
    random.shuffle(index)
    centroids = []
    for i in range(n):
        centroids.append(tf.Variable(sample[index[i]],dtype=tf.float32))
    # centroids = [tf.Variable(sample[index[i]] for i in range(n))]

    assignments = [tf.Variable(0) for i in range(len(sample))]


    centroidsPlace = tf.placeholder(shape=[2],dtype=tf.float32,name='centroidsPlace')
    assignmentPlace = tf.placeholder(shape=[],dtype=tf.int32,name='assignmentPlace')
    xSampleplace = tf.placeholder(shape=[None,2],dtype=tf.float32,name='xSampleplace')
    v1 = tf.placeholder(shape=[None],dtype=tf.float32,name='v1')
    v2 = tf.placeholder(shape=[None],dtype=tf.float32,name='v1')
    centroid_disPlace = tf.placeholder(shape = [None], dtype = tf.float32, name = 'centroid_distances')

    # 更新后的聚类点和分类
    cent_assigns = []
    for centroid in centroids:
        cent_assigns.append(tf.assign(centroid, centroidsPlace))
    cluster_assigns = []
    for assignment in assignments:
        cluster_assigns.append(tf.assign(assignment,assignmentPlace))


    mean_op = tf.reduce_mean(xSampleplace, 0)
    o_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(v1, v2))))

    cluster_assignment = tf.argmin(centroid_disPlace, 0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            flag1 = [np.array(assignments)[i].eval() for i in range(len(assignments))]

            for vec in range(len(sample)):
                vect = sample[vec]
                # 样本点到每一个聚类中心的距离
                distance = [sess.run(o_dist,feed_dict={v1:vect,v2:sess.run(centroid)}) for centroid in centroids]
                # 选取距离最小的
                assig = sess.run(cluster_assignment,feed_dict={centroid_disPlace:distance})

                sess.run(cluster_assigns[vec],feed_dict={assignmentPlace:assig})
            # 更新聚类点
            for j in range(n):
                assigned_vects = [sample[i] for i in range(len(sample)) if sess.run(assignments[i]) == j]
                new_location = sess.run(mean_op,feed_dict={xSampleplace:np.array(assigned_vects)})
                sess.run(cent_assigns[j], feed_dict={centroidsPlace: new_location})

            flag = [np.array(assignments)[i].eval() for i in range(len(assignments))]
            if flag == flag1:
                break

        print(cent_assigns)
        centroids = sess.run(centroids)
        assignments = sess.run(assignments)

        assignments = np.array(assignments)
        print(centroids)
        plt.scatter(sample[assignments==1,0],sample[assignments==1,1],c = 'y',marker='*' )
        plt.scatter(sample[assignments==0,0],sample[assignments==0,1],c = 'g',marker='^' )
        plt.scatter(centroids[0][0],centroids[0][1],c = 'r',marker='h')
        plt.scatter(centroids[1][0], centroids[1][1], c='r', marker='h')
        plt.show()
        # return centroids, assignments



def imgDispose(url=None):
    """
    从网上下载一张图片
    对图片大小，对比度，色度，饱和度，量度进行处理
    对图像进行翻转
    :param url: 图像地址
    :return:
    """
    # request.urlretrieve('http://www.jiningsoftware.com/public/upload/images/2018-01-22/5a659b9686340.jpg','./data/demo.jpg')
    img = Image.open('./data/image/demo.jpg')
    # img = img.resize((200, 200))
    # print(img.format)
    im = np.array(img)
    print(im.shape)
    # plt.imshow(im.reshape(657, 1920, 3))
    # plt.show()
    # print(img.size)
    # img.show()
    imgGray = img.convert('L')
    # imgGray.show()
    # 旋转90度
    imgR90 = img.rotate(90)
    # imgR90.show()
    # 水平翻转
    imgTr = img.transpose(Image.FLIP_LEFT_RIGHT)
    # imgTr.show()
    # 垂直翻转
    imgTb = img.transpose(Image.FLIP_TOP_BOTTOM)
    # imgTb.show()
    imgTrb = imgTb.transpose(Image.FLIP_LEFT_RIGHT)
    # imgT = img.transpose(PIL.Image.ROTATE_270)
    # imgTrb.show()

    #图片亮度
    enh_bri = ImageEnhance.Brightness(img)
    brightness = 1.5
    img_bright = enh_bri.enhance(brightness)
    # img_bright.show()

    # 图片对比度
    enh_con = ImageEnhance.Contrast(img)
    con = 1.5
    img_contrast = enh_con.enhance(con)
    # img_contrast.show()

    # 图片饱和度个（锐度）

    enh_sha = ImageEnhance.Sharpness(img)
    sharpness = 3.0
    image_sharped = enh_sha.enhance(sharpness)
    # image_sharped.show()

    # 图片色度
    enh_col = ImageEnhance.Color(img)
    color = 1.5
    image_colored = enh_col.enhance(color)
    # image_colored.show()


def loadDataCsv():
    """
    加载白酒数据
    :return: sample  特征数据集
    :return: labelSample  数据分类数据  one_hot
    """
    data = pd.read_csv('./data/winequality-white.csv',delimiter=';')
    sample = data.values[:,:-1]
    labelSample = data.values[:,-1]
    labelSample = pd.get_dummies(labelSample)
    return sample,labelSample



def wineWhite(sample,labelSample):
    """
    对白酒品质数据进行处理，并且使用神经网络学习算法，进行训练，并给出训练模型正确率
    数据集随机分为训练集60%，测试集20%，验证集20%
    绘制出每一个数据集的loss图像
    :param sample:   特征数据集
    :param labelSample: 分类数据集
    :return: None
    """
    # 超参
    trainStep = 2000
    learningRate = 2
    trainNum = len(sample)
    trainPrence = 0.6
    testPrecen = 0.2
    proviPrecen = 0.2

    #归一化
    sample =(sample - sample.min(0))/(sample.max(0) - sample.min(0))


    labelSample = np.array(labelSample,dtype=np.float32)

    #随机取训练集、测试集、验证集
    index = [ i for i in range(trainNum)]
    random.shuffle(index)

    trainIndex = index[:round(trainNum*trainPrence)]
    testIndex = index[round(trainNum*trainPrence):round(trainNum*(trainPrence+testPrecen))]
    proviIndex = index[round(trainNum*(trainPrence+testPrecen)):]

    trainSample = sample[trainIndex]

    trainLabel = labelSample[trainIndex]
    print(trainLabel.shape)
    testSample = sample[testIndex]
    testLable = labelSample[testIndex]
    proviSample = sample[proviIndex]
    proviLabel = labelSample[proviIndex]

    # 占位
    xPlace = tf.placeholder(shape=[None, 11], dtype=tf.float32, name='xPlace')
    labelPlace = tf.placeholder(shape=[None, 7], dtype=tf.float32, name='labelPlace')

    # 变量
    W = tf.Variable(tf.random_normal(shape=[11, 7]), dtype=tf.float32, name='W')

    b = tf.Variable(tf.random_normal(shape=[]), dtype=tf.float32, name='b')

    #预测模型
    pred =tf.add(tf.matmul(xPlace, W), b)
    #损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labelPlace, logits=pred))
    #准确率
    accur = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits=pred), axis=1), tf.argmax(labelPlace, axis=1)), tf.float32))
    #优化器
    my_opt = tf.train.AdamOptimizer(learningRate)
    trainProcess = my_opt.minimize(loss, name='trainProcess')

    losses_tr = []
    losses_te = []
    losses_pr = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #训练
        for i in range(trainStep):
            sess.run(trainProcess,feed_dict={xPlace:trainSample,labelPlace:trainLabel})
            losses_tr.append(sess.run(loss,feed_dict={xPlace:trainSample,labelPlace:trainLabel}))
            losses_te.append(sess.run(loss,feed_dict={xPlace:testSample,labelPlace:testLable}))
            losses_pr.append(sess.run(loss,feed_dict={xPlace:proviSample,labelPlace:proviLabel}))

        accues = sess.run(accur,feed_dict={xPlace:testSample,labelPlace:testLable})

        print('准确率为：%0.5f%% ' % (accues*100))

        linex = np.linspace(-3,5,trainStep)
        plt.plot(linex,losses_tr,'r-',label = 'train')
        plt.plot(linex,losses_te,'g--',label='test')
        plt.plot(linex,losses_pr,'y',label='prov')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    sample = loadData('./data/iris')
    #task1
    # kmeans
    # tfKmeans(sample,2)
    #task2
    # 图片处理
    imgDispose()
    # sample , label = loadDataCsv()
    # print(type(sample))
    #task3
    # wineWhite(sample,label)
