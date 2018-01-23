from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets('/path/to/MNIST_data',one_hot=True)

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE =500
BATCH_SIZE = 100

learningRate = 0.8
laerningRateDecay = 0.99

RegularizationRate = 0.0001
trainStep = 3000

def inference(input_tensor,avg_class, weight1,biases1,weight2,biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weight1)+biases1)
        return tf.matmul(layer1,weight2)+biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weight2))+avg_class.average(biases2)
def train(mnist):
    x = tf.placeholder(shape=[None,INPUT_NODE],dtype=tf.float32,name='x')
    y_ = tf.placeholder(shape=[None,OUTPUT_NODE],dtype=tf.float32,name='y')
    
    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev = 0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

np.round()