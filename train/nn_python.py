import numpy as np
import pandas as pd


# [10,10]
def nn_demo(x, y, layer, learning_Rate,loop=0):
    theta = []
    feature1 = x.shape[0]
    feature2 = y.shape[0]
    d = [feature1]
    d.extend(layer)
    d.extend([feature2])
    for i in range(1, len(d)):
        theta.append(np.random.randn(d[i], d[i - 1]))
    for o in range(1000):
        a = [x]
        for i in range(1, len(d)):
            a.append(1 / (1 + np.exp(-np.dot(theta[i - 1], a[i - 1]))))
        delta = [a[-1]-y]
        tmp = len(a)
        tmp_t = len(theta)
        for j in range(1, len(d)-1):
            delta.append(np.dot(theta[tmp_t - j].T, delta[-1]) * a[tmp - 1 - j] * (1 - a[tmp - 1 - j]))
        tmp_n = x.shape[1]
        len_d = len(delta)
        for k in range(len(theta)):
            theta[k] -= learning_Rate * np.dot(delta[len_d - 1 - k], a[k].T) / tmp_n
    return a[-1]

x = np.random.randn(6,5).T
y = [1,2,2,0,3,0]
y = pd.get_dummies(y).values.T


theta1 = np.random.randn(10,5)
theta2 = np.random.randn(10,10)
theta3 = np.random.randn(4,10)

for i in range(1):
    # 正向传播
    # a1 = x
    # a2 = 1/(1+np.exp(-np.dot(theta1,a1)))
    # a3 = 1/(1+np.exp(-np.dot(theta2,a2)))
    # a4 = 1/(1+np.exp(-np.dot(theta3,a3)))
    #
    # #后向反馈
    # delta4 = a4 -y
    # print(delta4)
    # delta3 = np.dot(theta3.T, delta4)*a3*(1-a3)
    # delta2 = np.dot(theta2.T, delta3)*a2*(1-a2)
    # print(delta2)
    # print(delta3)
    #
    # tmp = x.shape[1]
    # theta1 -=  np.dot(delta2,a1.T)/tmp
    # theta2 -=  np.dot(delta3,a2.T)/tmp
    #
    # theta3 -=  np.dot(delta4,a3.T)/tmp

    a4 = nn_demo(x,y,[10,10],1)
# print(a4)
print(np.argmax(a4,axis=0))




# nn_demo(x,y,[10,10],0.7)
# 4,6
# delta3 (10, 6)
# delta2 (10, 6)


# theta1 (10, 5)
# theta2 (10, 10)
# theta3 (4, 10)