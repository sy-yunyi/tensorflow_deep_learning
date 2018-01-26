import tensorflow as tf


def saveDemo():
    x1 = tf.placeholder(dtype=tf.float32,shape=[],name='x1')
    x2 = tf.placeholder(dtype=tf.float32,shape=[],name='x1')
    w = tf.Variable(tf.constant(2.),name='W')
    w2 = tf.Variable(tf.constant(2.), name='W')
    ytmp = tf.multiply(w,x1,name='ytemp')
    y = tf.add(ytmp,x2,name='y')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ypred = sess.run(y,feed_dict={x1:1,x2:2})
    saver = tf.train.Saver()
    saver.save(sess,'test/model')

def import_graph():
    sess = tf.Session()
    saver = tf.train.import_meta_graph('test/model.meta')
    saver.restore(sess,'test/model')
    graph = tf.get_default_graph()
    print(tf.global_variables())


if __name__ == '__main__':
    # saveDemo()
    import_graph()
