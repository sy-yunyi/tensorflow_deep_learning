import numpy as np
import tensorflow as tf
import random

max_sequence_length = 25
min_word_requency = 0
# 每个词对应的向量的长度
embedding_size = 50


def loadData(filePath):
    labels = []
    mail_data = []
    fr = open(filePath)
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        line = line.split('\t')
        labels.append(line[0])
        mail_data.append(line[1])

    labels = [0 if i== 'ham' else 1 for i in labels ]
    return labels,mail_data

# x = np.load('./data/mail_x.npy')
# print(x.shape)

def word2vec(mail_data):

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,min_word_requency)
    text_processed = np.array(list(vocab_processor.fit_transform(mail_data)))
    vocab_size = len(vocab_processor.vocabulary_)


    #生成每个词对应的向量
    embedding_mat = tf.Variable(tf.random_normal([vocab_size,50],-1.0,1.0))

    #生成每个句子的向量表示
    embedding_output = tf.nn.embedding_lookup(embedding_mat,text_processed)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        word_vec = sess.run(embedding_output)
    return word_vec,vocab_size


def cnn_func(x):
    # 最后一层输出的神经元个数
    output_layer = 1

    rnn_size = 256
    #初始化 rnn cell
    cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)

    #循环过程
    outputs, h = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)

    # 取最后一个状态，组建全连接
    W = tf.Variable(tf.truncated_normal([rnn_size, output_layer], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[output_layer]))
    pred = tf.add(tf.matmul(h, W), b)
    return pred


def classifiy_rnn(mail_data,labels):

    #超参
    learning_rate = 0.0001
    trainPercent = 0.7


    trainStep = 2000

    mail_vec,vocab_size = word2vec(mail_data)
    labels = np.array(labels).reshape(-1, 1)
    num_sample = mail_vec.shape[0]

    index = [i for i in range(mail_vec.shape[0])]
    random.shuffle(index)

    trainIndex = index[:round(num_sample * trainPercent)]
    testIndex = index[round(num_sample * trainPercent):]

    trainSample = mail_vec[trainIndex,:,:]
    trainLabel = labels[trainIndex]
    testSample = mail_vec[testIndex,:,:]
    testLabel = labels[testIndex]

    xPlace = tf.placeholder(shape=[None,max_sequence_length,embedding_size],dtype=tf.float32,name='xPlace')
    labelPlace = tf.placeholder(shape=[None,1],dtype=tf.float32,name='labelPlace')

    pred = cnn_func(xPlace)

    accur = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.nn.sigmoid(pred)),labelPlace),tf.float32))


    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labelPlace,logits=pred)

    myopt = tf.train.AdamOptimizer(learning_rate)

    trainProcess = myopt.minimize(loss,name='trainProcess')

    losses = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(trainStep):
            sess.run(trainProcess,feed_dict={xPlace:trainSample,labelPlace:trainLabel})
            curAcc = sess.run(accur,feed_dict={xPlace:testSample,labelPlace:testLabel})
            print(curAcc)






if __name__ == '__main__':
    filePath = './data/mail'
    labels, mail_data = loadData(filePath)
    classifiy_rnn(mail_data,labels)
