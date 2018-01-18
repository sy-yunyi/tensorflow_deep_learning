import tensorflow as tf
import pdb
import numpy as np
zero_tsr = tf.zeros([2,3])
ones_tsr = tf.ones([3,4])
filled_tsr = tf.fill([3,3],12.0)
constant = tf.constant([1,2,3])

zeros_similar = tf.zeros_like(constant)
ones_similar = tf.ones_like(constant)

linear_tsr = tf.linspace(start=0.0,stop = 1.0,num = 3)

intlger_seq_str = tf.range(start=6,limit = 15,delta=3)

randuif_str = tf.random_uniform([3,3],minval=0,maxval=3)

indentity_mat = tf.diag([1.0,1.0,1.0])
A = tf.truncated_normal([3,3])
B= tf.fill([3,3],5.0)
D = tf.convert_to_tensor(np.random.rand(3,3))

sess = tf.Session()
a = np.zeros([3,3])
b = np.ones([3,3])
# print(sess.run(a+b))

a = tf.convert_to_tensor(a)
b = tf.convert_to_tensor(b)
print(sess.run(constant))
print(sess.run(intlger_seq_str))
print(sess.run(tf.minimum(constant,intlger_seq_str)))
# with tf.Session() as sess :
    # pdb.set_trace()
    # print(sess.run(tf.matmul(B,indentity_mat)))
    # print(sess.run(D))

    # print(sess.run(tf.matrix_inverse(D)))

    # print('*'*5)
    # print(sess.run(ones_tsr))
    # print('*'*5)
    # print(sess.run(filled_tsr))
    # print('*'*5)
    # print(sess.run(constant))
    # print('*' * 5)
    # print(sess.run(zeros_similar))
    # print('*' * 5)
    # print(sess.run(ones_similar))
    # print('*' * 5)
    # print(sess.run(linear_tsr))
    # print('*' * 5)
    # print(sess.run(intlger_seq_str))
    # print('*' * 5)
    # print(sess.run(randuif_str))