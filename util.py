from sklearn import tree
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_train(f):
    with open(f, 'r') as fin:
        # Get rid of the first line
        fin.readline()
        data = np.array(list(csv.reader(fin))).astype(float)
    ids = data[:,0]
    features = data[:, 1:len(data[1,:])-1]
    labels = data[:, len(data[1,:])-1]
    fin.close()
    return ids, features, labels

def load_test(f):
    with open(f, 'r') as fin:
        # Get rid of the first line
        fin.readline()
        data = np.array(list(csv.reader(fin))).astype(int)
    ids = data[:,0]
    features = data[:, 1:len(data[1,:])]
    fin.close()
    return ids, features

def write_predictions(pred, f):
    with open(f, 'w') as fin:
        fin.write('Id,Prediction\n')
        for i in range(len(pred[:,0])):
            fin.write(str(int(pred[i,0])) + ',' + str(int(pred[i,1])) + '\n')
    fin.close()
