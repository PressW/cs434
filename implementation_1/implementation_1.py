import pandas as pd
import numpy as np
import math, random


col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
             'PTRATIO', 'B', 'LSTAT', 'MEDV']


train_file = 'housing_train.txt'
test_file = 'housing_test.txt'


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
    rand_list = [random.uniform(0.0, float(limit)) for i in range(size)]
    rand_col = np.array([])
    for rand in rand_list:




def generate_random_features(X, num_features):
    print(X[0:1])
    print(len(X))

    for i in range(num_features):
        rand_list = generate_random_column((random.uniform(25.0, 500.0)), len(X))
        new_col = [
        print(new_col[1:10])
        new_X = np.append(X, new_col.transpose(), 1)
        X = new_X

        print(X[0:2])







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



#problems_1_to_3()
#problem_4()

random_list = generate_random_column(356, 15)
X,Y = make_matrix(train_file)
#print(random_list)
generate_random_features(X, 3)



