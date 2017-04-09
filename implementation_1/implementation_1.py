import pandas as pd
import numpy as np
import math


col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
             'PTRATIO', 'B', 'LSTAT', 'MEDV']


train_file = 'housing_train.txt'
test_file = 'housing_test.txt'


def make_matrix(datafile, dummy=True):
    
    df = pd.read_csv(datafile, delim_whitespace=True, header=None, names=col_names)
    if dummy is True:
        df.insert(0, 'DUMMY', 1)
        X_names = ['DUMMY', 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
               'PTRATIO', 'B', 'LSTAT']
    else:
        X_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                   'PTRATIO', 'B', 'LSTAT']
    
    X = df.as_matrix(columns=X_names)
    Y = df.as_matrix(columns=['MEDV'])

    return X, Y



def compute_weight(X, Y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), Y)



def compute_sse(X,Y,W):
    actual_predicted_diff = []
    for index, matrix in enumerate(X):
        actual_predicted_diff.append(math.pow(Y[index] - sum([W[k] * matrix[k] for k in range(len(matrix))]), 2))
    
    return sum(actual_predicted_diff)







def problems_1_to_3():
    print("\n-------- With dummy column --------")

    X_train,Y_train = make_matrix(train_file)
    W = compute_weight(X_train,Y_train)
    train_sse = compute_sse(X_train, Y_train, W)

    print("Training SSE: ", train_sse)

    X_test,Y_test = make_matrix(test_file)
    test_sse = compute_sse(X_test, Y_test, W)

    print("Testing SSE: ", test_sse)


def problem_4():
    print("\n-------- Without dummy column --------")

    X_train,Y_train = make_matrix(train_file, dummy=False)
    W = compute_weight(X_train,Y_train)
    train_sse = compute_sse(X_train, Y_train, W)

    print("Training SSE: ", train_sse)

    X_test,Y_test = make_matrix(test_file, dummy=False)
    test_sse = compute_sse(X_test, Y_test, W)

    print("Testing SSE: ", test_sse, "\n")

problems_1_to_3()
problem_4()
