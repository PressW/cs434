import numpy as np
import pandas as pd
import math, random, sys, csv, operator
from math import exp



def main():
	print('Getting Data')
	X_train, Y_train, X_test, Y_test = get_data()

	print('starting part 1.1')
	# X_test[1] is just a random instance to pass the algorithm.
	neighbor_list, votes = nearest_neighbor(X_train, Y_train, X_test[1], 11)
	# Potential Example Usage
	total = 0
	pos = 0
	neg = 0
	for v in votes:
		total += 1
		if v == 1: pos += 1
		if v == -1: neg += 1
	if pos > neg: print("\nPositive classification with {0}%% confidence".format(pos/total))
	if neg > pos: print("\nNegative classification with {0}%% confidence".format(neg/total))



def get_data():
	df_train = pd.read_csv('knn_train.csv', header=None)
	X_train = df_train.as_matrix(columns=list(range(1,31)))
	Y_train = df_train.as_matrix(columns=[0])

	df_test = pd.read_csv('knn_test.csv', header=None)
	X_test = df_test.as_matrix(columns=list(range(1,31)))
	Y_test = df_test.as_matrix(columns=[0])

	return X_train, Y_train, X_test, Y_test



def nearest_neighbor(X, Y, x_instance, k):
	distances = []
	neighbors = []
	votes = []
	for i in range(len(X)):
		distances.append((X[i], get_distance(X, x_instance), Y[i]))
	distances.sort(key=operator.itemgetter(1))
	for i in range(k):
		# This has the actual nearest neighbors
		neighbors.append(distances[i][0])
		# This has the classification of those neighbors
		votes.append(distances[i][2])
	return neighbors, votes



def get_distance(X, x_i):
	return np.sqrt(np.sum([np.matmul(np.transpose(np.subtract(X[j], x_i)), np.subtract(X[j], x_i)) for j in range(len(X))]))


if __name__ == '__main__':
	main()
