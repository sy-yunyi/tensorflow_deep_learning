import tensorflow as tf
import pdb
#
# g1 = tf.Graph()
# with g1.as_default():
#     v = tf.get_variable('v',shape=[1],initializer=tf.zeros_initializer())
#
# g2 = tf.Graph()
# with g2.as_default():
#     v = tf.get_variable('v',shape=[1],initializer=tf.ones_initializer())
#
# with tf.Session(graph=g1) as sess:
#     tf.initialize_all_variables().run()
#     with tf.variable_scope("",reuse=True):
#         print(sess.run(tf.get_variable('v')))
#
# with tf.Session(graph=g2) as sess:
#     tf.initialize_all_variables().run()
#     with tf.variable_scope("",reuse=True):
#         print(sess.run(tf.get_variable('v')))
#
# sess = tf.Session()
# a = tf.constant([1.0,2.0],name='a')
# b = tf.constant([2.0,3.0],name='b')
# result = tf.add(a,b,name='add')
# with sess.as_default():
#     print(result.eval())
# # sess.run(result)
#
# weights = tf.Variable(tf.random_normal([2,3],stddev=2))

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x = tf.constant([[0.7,0.9]])

a = tf.matmul(x, w1)

y = tf.matmul(a, w2)

sess = tf.Session()
init_op = tf.initialize_all_variables()
sess.run(init_op)
# sess.run(w1.initializer)
# sess.run(w2.initializer)
pdb.set_trace()
print(sess.run(y))
sess.close()