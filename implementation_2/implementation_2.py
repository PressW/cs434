import pandas as pd
import numpy as np
import math, sys



def make_matrix(datafile):
    df = pd.read_csv(datafile, header=None)
    X = df.as_matrix(columns=range(256))
    Y = df.as_matrix(columns=[256])
    return X, Y



def compute_regresstion(X,Y_actual,eta):
    W = pd.DataFrame(np.zeros((1,256)))
    for j in range(100):
        D = pd.DataFrame(np.zeros((1,256)))
        for i in range(len(X)):
            Y_predicted = (1 / (1 + np.exp(np.matmul((np.negative(W)), np.transpose(X[i:(i+1)])))))
            err = Y_actual[i] - Y_predicted
            D = np.add(D, (err * X[i:(i+1)]))
        W = np.add(W, (eta * D))
    return W


                     
X,Y = make_matrix('usps-4-9-train.csv')
#print(Y)
W = compute_regresstion(X,Y,eta=0.00000002)
print(W.values)
