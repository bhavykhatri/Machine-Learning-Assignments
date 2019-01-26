#!/usr/bin/python
import sys
import os
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

FEATURE = 4096
CLASSES = 50
SEEN = 40
UNSEEN = 10
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

def compute_s(s):
    for i in range(UNSEEN):
        for j in range(SEEN):
            s[i][j] = np.dot(class_attributes_unseen[i], class_attributes_seen[j])

def normalize_s(s):
    for i in range(UNSEEN):
        s[i, :] = s[i, :]/s[i, :].sum()

def compute_unseen_mu(mu, s):
    for i in range(UNSEEN):
        mu[SEEN + i] = np.dot(s[i, :], mu[0:SEEN, :])

def prototype_prediction(X_test, mu):
    flag = np.zeros( ( X_test.shape[0], UNSEEN) )
    Y_pred = np.zeros(( X_test.shape[0], 1))
    for i in range(UNSEEN):
        flag[:, i] = LA.norm(X_test - mu[SEEN + i, :], axis =  1)
    Y_pred = np.argmin(flag, axis = 1)  +1
    return Y_pred

def accuracy(Y_pred, Y_test):
    return np.array(Y_pred  == Y_test, dtype=int).mean()

s = np.zeros((UNSEEN, SEEN))
mu = np.zeros((CLASSES, FEATURE) )
compute_mu(mu)
compute_s(s)
normalize_s(s)
compute_unseen_mu(mu, s)

Y_pred  = prototype_prediction(X_test, mu).reshape(X_test.shape[0], 1)

print ("Accuracy of prototype classifier is:", accuracy(Y_pred, Y_test)*100, "%" )

