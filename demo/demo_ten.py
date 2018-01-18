import tensorflow as tf

x = tf.placeholder(shape=[],dtype=tf.int32,name='x')

w = tf.Variable(tf.constant(10),name='w')
b = tf.Variable(tf.constant(5),name='b')

y = w*x +b

x_data = [i for i in range(3)]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for item in x_data:
        print(sess.run(y,feed_dict={x:item}))