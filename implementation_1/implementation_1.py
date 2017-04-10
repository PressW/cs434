import pandas as pd
import numpy as np
import math, random


col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
             'PTRATIO', 'B', 'LSTAT', 'MEDV']


train_file = 'housing_train.txt'
test_file = 'housing_test.txt'




# ------------ Helper Funcitons ------------ #

def make_matrix(datafile, dummy=True):

    df = pd.read_csv(datafile, delim_whitespace=True, header=None, names=col_names)
    if dummy is True:
        df.insert(0, 'DUMMY', 1)
        X_names = ['DUMMY']
        X_names.extend(col_names)
        X_names.remove('MEDV')
    else:
        X_names = []
        X_names.extend(col_names)
        X_names.remove('MEDV')

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



def generate_random_column(limit, size):
    return np.asarray([random.uniform(0.0, float(limit)) for i in range(size)])



def generate_random_features(X, num_features):
    for i in range(num_features):
        new_col = generate_random_column((random.uniform(25.0, 500.0)), len(X))
        new_X = np.c_[X,new_col]
        X = new_X
    return X




# ------------ Show Problem Results ------------ #

def problems_1_to_3():
    print("\n-------- With dummy column --------")

    X_train,Y_train = make_matrix(train_file)
    W = compute_weight(X_train,Y_train)
    print("\nW vector:\n", W, "\n")
    train_sse = compute_sse(X_train, Y_train, W)

    print("Training SSE: ", train_sse)

    X_test,Y_test = make_matrix(test_file)
    test_sse = compute_sse(X_test, Y_test, W)

    print("Testing SSE: ", test_sse)


def problem_4():
    print("\n-------- Without dummy column --------")

    X_train,Y_train = make_matrix(train_file, dummy=False)
    W = compute_weight(X_train,Y_train)
    print("\nW vector:\n", W, "\n")
    train_sse = compute_sse(X_train, Y_train, W)

    print("Training SSE: ", train_sse)

    X_test,Y_test = make_matrix(test_file, dummy=False)
    test_sse = compute_sse(X_test, Y_test, W)

    print("Testing SSE: ", test_sse)


def problem_5(num_iterations):
    print("\n\n----- Random Feature Generation -----")
    for i in range(num_iterations):
        rand = random.randint(1, 12)
        print("\n***** Iteration {0}: Creating {1} randomized features *****".format(i, rand))

        X_train,Y_train = make_matrix(train_file)
        X_train = generate_random_features(X_train, rand)
        W = compute_weight(X_train, Y_train)
        train_sse = compute_sse(X_train, Y_train, W)
        print("Training SSE {0}: ".format(i), train_sse)

        X_test,Y_test = make_matrix(train_file)
        X_test = generate_random_features(X_test, rand)
        test_sse = compute_sse(X_test, Y_test, W)
        print("Testing SSE {0}: ".format(i), test_sse)
    print("\n")



if __name__ == "__main__":
    problems_1_to_3()
    problem_4()
    problem_5(5)
