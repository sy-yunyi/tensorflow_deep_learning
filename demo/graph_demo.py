import tensorflow as tf

a = tf.constant(2,name='a')
b = tf.constant(3,name='b')
c = tf.constant(4,name='c')
d = tf.constant(5,name='d')

add1 = tf.add(a,b,name='add1')
mul1 = tf.multiply(c,d,name='mul1')

add2 = tf.add(add1,mul1,name='add2')
mul2 = tf.multiply(add1,mul1,name='mul2')

output1 = tf.div(add2,mul2,name='output1')

with tf.name_scope('scope1'):
    output2 = tf.square(output1,name='output2')
    output3 = tf.square(output2,name='output3')
output4 = tf.add(output2,output3,name='output4')
with tf.Session() as sess:
    my_graph = tf.summary.FileWriter('E:\\tensorboardlog\\',sess.graph)
    sess.run(output1)