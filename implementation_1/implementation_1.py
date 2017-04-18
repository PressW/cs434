import pandas as pd
import numpy as np
import math, random, sys


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



def compute_lambda_weight(X, Y, lmbda):
    return np.matmul(np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(X), X), (lmbda * np.identity(len(np.matmul(np.transpose(X), X)))))), np.transpose(X)), Y)



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

    print("Testing SSE: ", test_sse, "\n")


def problem_4():
    print("\n-------- Without dummy column --------")

    X_train,Y_train = make_matrix(train_file, dummy=False)
    W = compute_weight(X_train,Y_train)
    print("\nW vector:\n", W, "\n")
    train_sse = compute_sse(X_train, Y_train, W)

    print("Training SSE: ", train_sse)

    X_test,Y_test = make_matrix(test_file, dummy=False)
    test_sse = compute_sse(X_test, Y_test, W)

    print("Testing SSE: ", test_sse, "\n")


def problem_5():
    print("\n\n----- Random Feature Generation -----")
    #for i in range(num_iterations):
    rands = [1, 2, 4, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100, 150, 250, 500]
    for rand in rands:
        #rand = random.randint(1, 35)
        #print("\n***** Iteration {0}: Creating {1} randomized features *****".format(i, rand))
        print("\n***** Creating {0} randomized features *****".format(rand))

        X_train,Y_train = make_matrix(train_file)
        X_train = generate_random_features(X_train, rand)
        W = compute_weight(X_train, Y_train)
        train_sse = compute_sse(X_train, Y_train, W)
        #print("Training SSE {0}: ".format(i), train_sse)
        print("Training SSE: ", train_sse)

        X_test,Y_test = make_matrix(test_file)
        X_test = generate_random_features(X_test, rand)
        test_sse = compute_sse(X_test, Y_test, W)
        #print("Testing SSE {0}: ".format(i), test_sse)
        print("Testing SSE: ", test_sse)
    print("\n")


def problem_6():
    print("\n\n----- Lambda Weight Calculations -----")
    X_train,Y_train = make_matrix(train_file)
    X_test,Y_test = make_matrix(test_file)

    values = [0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 25, 50, 100, 250, 500, 1000, 5000, 10000, 100000]
    for value in values:
        print("\n***** Lambda value {0} *****".format(value))
        W = compute_lambda_weight(X_train, Y_train, value)
        print("\nW vector:\n", W, "\n")
        print("\nW norm: ", np.linalg.norm(W, ord=2))
        train_sse = compute_sse(X_train, Y_train, W)
        print("Training SSE: ", train_sse)
        test_sse = compute_sse(X_test, Y_test, W)
        print("Testing SSE: ", test_sse)
    print("\n")




if __name__ == "__main__":
    try:
        arg = int(sys.argv[1])
        if arg >= 1 and arg <=3:
            problems_1_to_3()
        elif arg == 4:
            problem_4()
        elif arg == 5:
            problem_5()
        elif (arg >= 6) and (arg <= 8):
            problem_6()
        elif len(sys.argv) == 1:
            problems_1_to_3()
            problem_4()
            problem_5()
            problem_6()
        else:
            print("\nUSAGE: python implementation_1.py <question>")
            print("\n<question>\tdefaults to printing all questions if not parameter passed\n\n")
    except:
        print("\nUSAGE: python implementation_1.py <question>")
        print("\n<question>\tdefaults to printing all questions if not parameter passed\n\n")
