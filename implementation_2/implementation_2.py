import pandas as pd
import numpy as np
import math, sys



def make_matrix(datafile):
    df = pd.read_csv(datafile, header=None)
    df.insert(0, "dummy", 1)
    cols = df.columns.tolist()
    cols = cols[:-1]
    X = df.as_matrix(columns=cols)
    Y = df.as_matrix(columns=[256])
    return X, Y



def compute_regression(X_train,Y_train_actual,X_test,Y_test_actual,eta):
    iters = 0
    correct_train = 0
    total_train = 0
    correct_test = 0
    total_test = 0
    accuracy_train = []
    accuracy_test = []

    W = pd.DataFrame(np.zeros((1,257)))
    norm = 9999999
    # for j in range(10):
    while norm > 1000:
        iters += 1
        D = pd.DataFrame(np.zeros((1,257)))
        for i in range(len(X_train)):
            prob_train_one = (1 / (1 + np.exp(np.matmul((np.negative(W)), np.transpose(X_train[i:(i+1)])))))
            prob_train_zero = 1 - prob_train_one
            ratio = prob_train_one / prob_train_zero
            if ratio > 1:
                prediction = 1
            else:
                prediction = 0
            if (prediction == Y_train_actual[i]):
                correct_train = correct_train + 1
            err = Y_train_actual[i] - prob_train_one
            D = np.add(D, (err * X_train[i:(i+1)]))
            total_train = total_train + 1

        for i in range(len(X_test)):
            prob_test_one = (1 / (1 + np.exp(np.matmul((np.negative(W)), np.transpose(X_test[i:(i+1)])))))
            prob_test_zero = 1 - prob_test_one
            ratio = prob_test_one / prob_test_zero
            if ratio > 1:
                prediction = 1
            else:
                prediction = 0
            if (prediction == Y_test_actual[i]):
                correct_test = correct_test + 1
            total_test = total_test + 1

        W = np.add(W, (eta * D))
        norm = np.linalg.norm(D, ord=2)
        accuracy_train.append(correct_train / total_train)
        accuracy_test.append(correct_test / total_test)
    return W, norm, iters, accuracy_train, accuracy_test


def problem_2():
    train_X,train_Y = make_matrix('usps-4-9-train.csv')
    test_X,test_Y = make_matrix('usps-4-9-test.csv')
    train_W, train_norm, train_count, train_acc, test_acc = compute_regression(train_X,train_Y,test_X,test_Y,eta=0.0000002)
    print("\n", train_norm, "\n")
    print("ITERATIONS: ", train_count, "\n")
    print("TRAINING ACCURACIES: \n")
    for i in range(train_count):
        print(train_acc[i])
    print("TESTING ACCURACIES: \n")
    for i in range(train_count):
        print(test_acc[i])

problem_2()
