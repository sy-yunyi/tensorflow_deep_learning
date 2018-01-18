import tensorflow as tf
from tensorflow.python.framework import ops

# sess = tf.Session()
#
# #获取默认图
# print(tf.get_default_graph())
# # 重置默认图
# ops.reset_default_graph()
# print(tf.get_default_graph())


# 定义一个图
g1 = tf.Graph()
g2 = tf.Graph()

#将图1设置为默认图，则在下面的with语句块中，所做操作为在图1 中的操作
with g1.as_default():
    a = tf.constant(10,name='a')
    b = tf.constant(20,name='b')
    c = tf.add(a,b,name='c')
# 图2
with g2.as_default():
    a = tf.constant(5,name='a')
    b = tf.constant(4,name='b')
    d = tf.add(a,b,name='d')


# 为session指定图
with tf.Session(graph=g1) as sess:
    my_graph = tf.summary.FileWriter('E:\\tensorboardlog\\',sess.graph)
    print(sess.run(c))
    print(c.eval())

with tf.Session(graph=g2) as sess:
    my_graph = tf.summary.FileWriter('E:\\tensorboardlog\\', sess.graph)
    print(sess.run(d))
    print(d.eval())