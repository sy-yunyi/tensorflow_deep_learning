import tensorflow as tf
import numpy as np
import os
import pickle
def loadData(filePath):
    data_set = []
    labels_set = []
    fileName = 'data_batch_'
    for i in range(1,6):
        fr = open(filePath+'\\'+fileName+str(i),'rb')
        dict = pickle.load(fr,encoding='bytes')
        data_set.extend(dict.pop(b'data'))
        labels_set.extend(dict.pop(b'labels'))
    fr = open(filePath+'\\'+'test_batch','rb')
    dict = pickle.load(fr,encoding='bytes')
    data_set_test = dict.pop(b'data')
    labels_set_test = dict.pop(b'labels')
    return data_set,labels_set,data_set_test,labels_set_test















if __name__ == '__main__':
    filePath = r'C:\Users\Administrator\Desktop\deeplearning\code\SampleData\cifar-10-batches-py'
    loadData(filePath)