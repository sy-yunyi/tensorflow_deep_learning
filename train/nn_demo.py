import numpy as np
import pandas as pd
x = np.random.randn(10,5)
y = [1,2,2,0,3,1,3,3,2,0]
y = pd.get_dummies(y)

weight1 = np.random.randn(10,5)
weight2 =np.random.randn(10,10)
weight3 = np.random.randn(4,10)

layer1 = np.dot(weight1,np.transpose(x))
print(layer1.shape)
layer1_a = 1/(1+np.exp(-layer1))
print(layer1_a.shape)
layer2 = np.dot(weight2,layer1_a)
layer2_a = 1/(1+np.exp(-layer2))
target = np.dot(layer2_a,np.transpose(weight3))
target_a = 1/(1+np.exp(-target))

# print(weight3.shape)
delat4 = (target_a - y).values
# print(delat4)
delat3 = np.dot(np.transpose(weight3),np.transpose(delat4)) * layer2_a *(1-layer2_a)
print(delat3.shape)
delat2 =np.dot(np.transpose(weight2),delat3) * layer1_a *(1-layer1_a)
print(delat2.shape)
# weight1 = weight1 - layer1_a[:,:] * np.transpose(delat2)
# weight2 = weight2 - layer2_a * delat3
# weight3 = weight3 - target_a * delat4


