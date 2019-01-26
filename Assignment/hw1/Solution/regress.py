#!/usr/bin/python
import sys
import os
import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import matplotlib.pyplot as plt

FEATURE = 4096
CLASSES = 50
SEEN = 40
UNSEEN = 10
ATTRIBUTE = 85
DATASET_FOLDER = sys.argv[1]

X_test = np.load(os.path.join(DATASET_FOLDER, 'Xtest.npy'), encoding = 'latin1')
Y_test=np.load(os.path.join(DATASET_FOLDER, 'Ytest.npy'), encoding = 'latin1')
X_seen = np.load(os.path.join(DATASET_FOLDER, 'X_seen.npy'), encoding = 'latin1')
class_attributes_seen=np.load(os.path.join(DATASET_FOLDER, 'class_attributes_seen.npy'), encoding = 'latin1')
class_attributes_unseen=np.load(os.path.join(DATASET_FOLDER, 'class_attributes_unseen.npy'), encoding = 'latin1')

def compute_mu(mu):
    for i in range(SEEN):
        single_class_data = X_seen[i]
        mu[i] = np.mean(single_class_data, axis = 0)

def compute_W(W, mu, As, lamb):
    W = np.matmul(LA.inv( np.matmul(np.transpose(As),As) +  lamb*np.identity(ATTRIBUTE))  ,np.matmul( np.transpose(As),mu[0:SEEN, :] ))
    return W

def compute_unseen_mu(W, mu):
    mu[SEEN:CLASSES, :] = np.transpose(np.matmul(np.transpose(W),np.transpose(class_attributes_unseen)))

def prototype_prediction(X_test, mu):
    flag = np.zeros( ( X_test.shape[0], UNSEEN) )
    Y_pred = np.zeros(( X_test.shape[0], 1))
    for i in range(UNSEEN):
        flag[:, i] = LA.norm(X_test - mu[SEEN + i, :], axis =  1)
    Y_pred = np.argmin(flag, axis = 1)  +1
    return Y_pred

def accuracy(Y_pred, Y_test):
    return np.array(Y_pred  == Y_test, dtype=int).mean()

def prediction_for_lambdas( lamb):
    acc = []
    for cell in lamb:
        W = np.zeros((ATTRIBUTE, FEATURE))
        W = compute_W(W, mu, class_attributes_seen, cell)
        compute_unseen_mu(W, mu)
        Y_pred = prototype_prediction(X_test, mu).reshape(X_test.shape[0], 1)
        acc.append(accuracy(Y_pred, Y_test)*100)
        print ("Accuracy of prototype classifier for lambda", cell, "is:", accuracy(Y_pred, Y_test)*100, "%" )
    return acc

mu = np.zeros((CLASSES, FEATURE))
compute_mu(mu)
lamb = [0.01, 0.1, 1, 10, 20, 50, 100]
acc_lamb = []
acc_lamb = prediction_for_lambdas( lamb)

print ("Maximum Accuracy is achieved for lambda = ", lamb[np.argmax(np.array(acc_lamb))])

plt.figure()
plt.plot(lamb, acc_lamb, 'go--')
plt.xlabel('Value of lambda')
plt.ylabel('Accuracy')
plt.show()
