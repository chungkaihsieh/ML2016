#!/usr/bin/env python
import numpy as np 
import pandas as pd
import scipy
import matplotlib.pyplot as plt 
import math
import csv
import time
import random
import sys

def read_data( testing_data_name ):
	'''
	read data
	'''

	# #Testing Data
	#read data
	columns_name = ['data_id']
	for i in range(57):
		columns_name.append(str(i))

	testing_data = pd.read_csv(testing_data_name, sep=',', encoding='latin1', names=columns_name)

	columns_select = []
	for i in range(57):
		columns_select.append(str(i))

	testing_set = testing_data.as_matrix(columns=columns_select).astype(dtype='float32')

	return testing_set


'''
Algorithm
'''

def rss(array):
	return np.sqrt(array.T.dot(array))


#usage: 
#data_set=numpy.array([])


def feature_normalize(data_set):
	tmp = data_set
	for i in range(np.shape(tmp)[1]):
		tmp[: , i] = ( tmp[:,i] - np.mean(tmp[:,i])) / np.std(tmp[:,i])

	return tmp

def feature_normalize_2(data_set):
	tmp = data_set
	for i in range(np.shape(tmp)[1]):
		tmp[: , i] = ( tmp[:,i] - np.mean(tmp[:,i])) 

	return tmp


def sigmoid(X):

	den = 1.0 + np.exp (-1.0 * X)
	d = 1.0 / den

	return d  





def testing(X,theta):
	X_tmp = X
	n = X.shape[0]
	bias_ones = np.ones( (n,1) , dtype='float32')
	X_tmp = np.append(X, bias_ones, axis=1)
	predictions_tmp = sigmoid( X_tmp.dot(theta) )
	predictions = np.rint( predictions_tmp ).astype(int)
	return  predictions



def write_file(predictions,file_name):
	predictions.shape = (len(predictions) , 1)
	writer = open(file_name,"w")
	writer.write("id,label\n")
	for i in range(len(predictions)):
		str_tmp = str( i+1 )+"," + str(predictions[i][0]) + "\n"
		# print(str_tmp)
		writer.write(str_tmp)		
	writer.close()


if __name__ == '__main__':


	start_time = time.time()

	model_name = sys.argv[1]
	testing_data_name = sys.argv[2]
	prediction_name = sys.argv[3]

	(testing_set) = read_data ( testing_data_name )
	testing_set_normalized = feature_normalize( testing_set )

	#load parameter
	theta_name = model_name+".npy"
	theta = np.load(theta_name)

	predictions = testing(testing_set_normalized,theta)


	# testing_set = (testing_set)	
	write_file(predictions, prediction_name)

	end_time = time.time()

	min_ = int(  (end_time - start_time)/60  )
	sec_ = (end_time - start_time)%60

	print("Consume : " + str(min_) + "mins " + str(sec_) + "secs" )




	# # print("starting point of  theta " + str(theta_tmp))

	# '''
	# Draw the cost function
	# '''
	# #gradient descent
	# plt.figure(1)
	# plot1 = plt.plot(range(len(cost_training_history[0:])) , cost_training_history[0:],'ro' ,label='$training$')
	# plot2 = plt.plot(range(len(cost_training_history[0:])) , cost_validation_history[0:],'g--',label ='$Validation$')

	# plt.title('Linear Regression')
	# plt.title('LR:' + str(learning_rate), loc='left')
	# plt.title('epochs :' + str(epochs), loc='right')
	# plt.ylabel('Cost')
	# plt.xlabel('epochs')
	# plt.xlim([0,len(cost_training_history[0:])])
	# plt.legend()
	# plt.show(block=True)

