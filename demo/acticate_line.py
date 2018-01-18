import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

x_vals = np.linspace(start=-5,stop=5,num=100)
y_vals = [x+2 for x in x_vals]
sess = tf.Session()

# plt.plot(x_vals,y_vals)

y_vals =[0 if x<0 else x for x in x_vals]
# plt.plot(x_vals,y_vals)

y_vals =[1/(1+np.exp(-x)) for x in x_vals ]
# plt.plot(x_vals,y_vals)

y_vals = [(np.exp(-x)-np.exp(x))/(np.exp(-x)+np.exp(x)) for x in x_vals]
# plt.plot(x_vals,y_vals)

# plt.show()

y_linear = sess.run(tf.add(2*x_vals,2))
y_relu = sess.run(tf.nn.relu(x_vals))
y_sigmoid = sess.run(tf.nn.sigmoid(x_vals))
y_tanh = sess.run(tf.nn.tanh(x_vals))

# plt.plot(x_vals, y_linear, 'r-', label = 'Linear')
# plt.plot(x_vals, y_relu, 'b-', label = 'Relu')
plt.plot(x_vals, y_sigmoid, 'y-', label = 'Sigmoid')
plt.plot(x_vals, y_tanh, 'g-', label = 'Tanh')
plt.legend(loc='upper left')

plt.show()