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

def normalize_train_standard_score(ftrain, f):
    ids, features, labels = load_train(ftrain)
    with open(f, 'w') as fin:
        fin.write('First row doesn\'t count\n')
        for i in range(len(features)):
            temp = features[i,:]
            mu = np.mean(temp)
            stdev = np.std(temp)
            temp2 = ''
            for j in range(len(temp)):
                temp2 = temp2 + str((temp[j]*1.0 - mu) / stdev)
                temp2 = temp2 + ','
            temp2 = temp2[:len(temp2)-1] + '\n'
            fin.write(temp2)
    fin.close()

def normalize_test_standard_score(ftrain, f):
    ids, features = load_test(ftrain)
    with open(f, 'w') as fin:
        fin.write('First row doesn\'t count\n')
        for i in range(len(features)):
            temp = features[i,:]
            mu = np.mean(temp)
            stdev = np.std(temp)
            temp2 = ''
            for j in range(len(temp)):
                if stdev != 0:
                    temp2 = temp2 + str((temp[j]*1.0 - mu) / stdev)
                    temp2 = temp2 + ','
                else:
                    temp2 = temp2 + '1,'
            temp2 = temp2[:len(temp2)-1] + '\n'
            fin.write(temp2)
    fin.close()
