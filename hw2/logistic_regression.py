#!/usr/bin/env python
import numpy as np 
import pandas as pd
import scipy
import matplotlib.pyplot as plt 
import math
import csv
import time
import random

def read_data():
	'''
	read data
	'''
	#Training Data
	#read data
	columns_name = ['data_id']
	for i in range(57):
		columns_name.append(str(i))
	columns_name.append('label')
	training_data = pd.read_csv('data/spam_train.csv', sep=',' , encoding='latin1' , names=columns_name)

	#select feature
	columns_select = []
	for i in range(57):
		columns_select.append(str(i))
	training_feature = training_data.as_matrix(columns=columns_select).astype(dtype='float32')
	columns_select  = ['label']
	training_yhat     = training_data.as_matrix(columns=columns_select ).astype(dtype='float32')

	print(training_feature)
	print("training_feature.shape" + str(training_feature.shape))
	print(training_yhat)
	print("training_yhat.shape" + str(training_yhat.shape))

	training_feature_total = training_feature
	training_yhat_total = training_yhat

	validation_percent = 1/10.0
	training_percent = 1 - validation_percent 
	num_training = int(training_percent * training_feature.shape[0])
	indices = np.random.permutation(training_feature.shape[0])
	training_idx,validation_idx = indices[:num_training], indices[num_training:]

	training_feature ,validation_feature = training_feature[training_idx,:], training_feature[validation_idx,:]
	training_yhat ,validation_yhat = training_yhat[training_idx,:], training_yhat[validation_idx,:]

	print("training_feature.shape" + str(training_feature.shape))
	print("validation_feature.shape" + str(validation_feature.shape))
	print("training_yhat.shape" + str(training_yhat.shape))
	print("validation_yhat.shape" + str(validation_yhat.shape))


	# #Testing Data
	#read data
	columns_name = ['data_id']
	for i in range(57):
		columns_name.append(str(i))

	testing_data = pd.read_csv('data/spam_test.csv', sep=',', encoding='latin1', names=columns_name)

	columns_select = []
	for i in range(57):
		columns_select.append(str(i))

	testing_set = testing_data.as_matrix(columns=columns_select).astype(dtype='float32')
	print(testing_set)
	print("testing_set.shape" + str(testing_set.shape))

	print("training_feature_total.shape" + str(training_feature_total.shape) )
	print("training_feature.shape" + str( training_feature.shape) )
	print("validation_feature.shape" + str( validation_feature.shape) )

	return training_feature_total, training_yhat_total , training_feature , training_yhat, validation_feature, validation_yhat ,testing_set


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


def compute_accuracy(X, y, theta):
	predictions = np.rint( sigmoid( X.dot(theta) ) ).astype(int)
	tmp = predictions
	for i in range(len(predictions)):
		# print("predictions[i]:" + str(predictions[i])   +" y[i]:" + str(y[i]))
		if( predictions[i] == y[i]):
			tmp[i] = 1
		else:
			tmp[i] = 0

	accurate_num = np.sum(tmp)
	accuracy = float(accurate_num)/len(predictions)
	print("[accurate:]" + str(accurate_num) + "/" + str(len(predictions)) + "  [accuracy:]" + str(float(accurate_num)/len(predictions))      )      

	return accuracy


#usage g_history = k*epochs  , k: number of parameters
def compute_adagrad_coeff(g_history):
	adagrad_coeff = np.sqrt( np.diag( g_history.dot(g_history.T )))
	return adagrad_coeff 

#compute cost value
def compute_cost(X ,y ,theta):
	# m : data number
	m = X.shape[0]
	theta = np.reshape(theta, (len(theta),1 ))

	# print("first component")
	# print(- np.transpose(y).dot(  np.log( sigmoid(X.dot(theta)) )) )

	# print("s2-1")
	# print(np.transpose( 1- y ))
	# print("s2-2")
	# print(sigmoid(X.dot(theta)))
	# print("2-2-1")
	# print(X)
	# print("2-2-2")
	# print(theta)
	# print("s2-3")
	# print(1 - sigmoid(X.dot(theta)))
	# print("s2-4")
	# print(np.log( 1 - sigmoid(X.dot(theta)) ))
	# print("second component")
	# print(- np.transpose( 1- y ).dot( np.log( 1 - sigmoid(X.dot(theta)) ) ))

	# J : cost 
	small_value = 10**-6
	J = (1.0/m) * ( - np.transpose(y).dot(  np.log( sigmoid(X.dot(theta)) + small_value )  )   - np.transpose( 1- y ).dot( np.log( 1 - sigmoid(X.dot(theta))  + small_value  )   )     )

	return J[0][0]


def compute_grad(X, y ,theta ):
	m = X.shape[0]
	theta = np.reshape(theta, (len(theta)),1 )
	h = sigmoid(X.dot(theta))

	# m: number of data, k number of feature
	#grad: 1xk
	# grad = np.transpose( (1.0/m)*(h - y) ).dot(X)
	grad = ( - 1.0/m) * np.transpose( (y -  h) ).dot(X)

	return grad

#*************************
#-----Gradient (adagrad)-----
#*************************
 
def gradient_descent(X, y, X_validation, y_validation, theta, learning_rate, epochs):
	#-------------------------------------------------
	# Performs gradient descent to learn theta
	# by taking epochs gradient steps with learning_rate
	#-------------------------------------------------
	X_tmp = X
	n = X.shape[0]
	bias_ones = np.ones( (n,1) , dtype='float32')
	X_tmp = np.append(X, bias_ones, axis=1)

	X_validation_tmp = X_validation
	n_validation = X_validation.shape[0]
	bias_ones = np.ones( (n_validation,1) , dtype='float32')
	X_validation_tmp = np.append(X_validation, bias_ones, axis=1)

	g_history = np.zeros( [len(theta), epochs] )
	cost_training_history = np.zeros(shape=(epochs, 1))
	cost_validation_history = np.zeros(shape=(epochs, 1))

	# print("X_validation.shape"  + str(X_validation.shape))
	# print("X_validation_tmp.shape"  + str(X_validation_tmp.shape))

	for i in range(epochs):
		predictions = X_tmp.dot(theta)
		theta_size = theta.shape[0]


	
		g = compute_grad(X_tmp, y, theta)[1,:]

		g_history[:, i ] = g
		adagrad_coeff = compute_adagrad_coeff(g_history)
		g = g.reshape( g.shape[0] , 1)
		adagrad_coeff = adagrad_coeff.reshape(adagrad_coeff.shape[0] , 1)

		theta = theta - learning_rate * g/adagrad_coeff

		print("iteration numbers : " + str(i))
		# print("theta : " + str(theta))
		print(" [cost] - training data   : " + str( compute_cost(X_tmp, y, theta)))
		compute_accuracy(X_tmp, y, theta)
		print(" [cost] - validation data : " + str( compute_cost(X_validation_tmp, y_validation, theta)))
		compute_accuracy(X_validation_tmp, y_validation, theta)
		cost_training_history[i, 0] = compute_cost(X_tmp, y, theta)
		cost_validation_history[i,0] = compute_cost(X_validation_tmp,y_validation,theta)
		validation_accuracy =  compute_accuracy(X_validation_tmp, y_validation, theta)

	return theta, cost_training_history, cost_validation_history , validation_accuracy

def gradient_descent_minibatch(X, y, X_validation, y_validation, theta, learning_rate, epochs,batch_size):
	#-------------------------------------------------
	# Performs gradient descent to learn theta
	# by taking epochs gradient steps with learning_rate
	#-------------------------------------------------
	X_tmp = X
	n = X.shape[0]
	bias_ones = np.ones( (n,1) , dtype='float32')
	X_tmp = np.append(X, bias_ones, axis=1)

	X_validation_tmp = X_validation
	n_validation = X_validation.shape[0]
	bias_ones = np.ones( (n_validation,1) , dtype='float32')
	X_validation_tmp = np.append(X_validation, bias_ones, axis=1)

	g_history = np.zeros( [len(theta), epochs] )
	cost_training_history = np.zeros(shape=(epochs, 1))
	cost_validation_history = np.zeros(shape=(epochs, 1))

	# print("X_validation.shape"  + str(X_validation.shape))
	# print("X_validation_tmp.shape"  + str(X_validation_tmp.shape))

	for i in range(epochs):

		for k in range( (X_tmp.shape[0] / batch_size) ):

			X_tmp_minibatch = X_tmp[k*batch_size: k*batch_size + batch_size, : ]
			y_minibatch = y[k*batch_size: k*batch_size + batch_size, : ]

			predictions = X_tmp_minibatch.dot(theta)
			theta_size = theta.shape[0]


			g = compute_grad(X_tmp_minibatch, y_minibatch, theta)[1,:]

			g_history[:, i ] = g
			adagrad_coeff = compute_adagrad_coeff(g_history)
			g = g.reshape( g.shape[0] , 1)
			adagrad_coeff = adagrad_coeff.reshape(adagrad_coeff.shape[0] , 1)


			theta = theta - learning_rate * g/adagrad_coeff
			# theta = theta - learning_rate * g

		print("iteration numbers : " + str(i))
		# print("theta : " + str(theta))
		training_accuracy =  compute_accuracy(X_tmp, y, theta)
		# print(" [m cost] - training data   : " + str( compute_cost(X_tmp, y, theta)))
		validation_accuracy =  compute_accuracy(X_validation_tmp, y_validation, theta)
		# print(" [m cost] - validation data : " + str( compute_cost(X_validation_tmp, y_validation, theta)))


		cost_training_history[i, 0] = compute_cost(X_tmp, y, theta)
		cost_validation_history[i,0] = compute_cost(X_validation_tmp,y_validation,theta)

	return theta, cost_training_history, cost_validation_history ,training_accuracy, validation_accuracy


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

	training_accuracy_th  = 0.89
	validation_accuracy_th = 0.93

	for k in range(1000):

		print("**************************")
		print("**************************")
		print("****  Data Changes  ****")
		print("**************************")
		print("**************************")
		# time.sleep(1)

		(training_feature_total, training_yhat_total, training_feature , training_yhat, validation_feature, validation_yhat ,testing_set) = read_data()
		# training_feature_total_normalized = feature_normalize(training_feature_total) 
		# training_feature_normalized = feature_normalize(training_feature) 
		# validation_feature_normalized = feature_normalize(validation_feature) 
		# testing_set_normalized = feature_normalize(testing_set)

		training_feature_total_normalized = feature_normalize(training_feature_total) 
		training_feature_normalized = feature_normalize(training_feature) 
		validation_feature_normalized = feature_normalize(validation_feature) 
		testing_set_normalized = feature_normalize(testing_set)


		# time.sleep(10)
		# training_feature_normalized_2 = feature_normalize_2(training_feature) 
		# validation_feature_normalized_2 = feature_normalize_2(validation_feature) 
		# testing_set_normalized_2 = feature_normalize_2(testing_set)

		# training_feature_normalized_2 = (training_feature) 
		# validation_feature_normalized_2 = (validation_feature) 
		# testing_set_normalized_2 = (testing_set)


		'''
		Test for all method
		'''
		for try_parameters in range(5):

			mu, sigma = 0, 0.001
			feature_num = 57
			theta = sigma * np.random.randn(1, feature_num +1).T + mu
			# print(theta.shape)
			# theta = np.random.uniform(-1,1, feature_num + 1)
			# theta = theta.reshape(feature_num+1,1)
			# print(theta.shape)
			# theta = np.zeros([1,feature_num+1]).astype(dtype='float32').T
			# theta = np.zeros([1,10]).T
			theta_tmp = theta
			# theta = (np.random.normal(mu, sigma,10)).T
			# theta = (np.random.rand(1,10).T - 0.5)*2

			epochs = 150
			epochs_minibatch  = 20
			batch_size = 15

			learning_rate = 0.02
			learning_rate_minibatch = 0.0005

			# print(training_set)
			# print(training_feature)
			# print(training_yhat)
			# gradient_descent(training_feature, training_yhat, theta, 0.0001,1500)

			# training_feature = feature_normalize(training_feature)
			# validation_feature = feature_normalize(validation_feature)
			# testing_set = feature_normalize(testing_set)
			print("*********** main *********")

			# training_feature = (training_feature)
			# validation_feature = (validation_feature) 



			'''
			normalized feature
			'''
			print("**************************")
			print("**************************")
			print("***** Normalized 1 ******")
			print("**************************")
			print("**************************")
			# time.sleep(0.5)
			# epochs = 50
			# learning_rate = 0.009
			# (theta_after,cost_training_history, cost_validation_history ,validation_accuracy) = gradient_descent(training_feature_normalized, training_yhat,validation_feature_normalized,validation_yhat, theta, learning_rate ,epochs)
			# if( validation_accuracy >= accuracy_th):
			# 	print("[normalized] method 1")
			# 	print("sigma:" + str(sigma) + "  learning_rate_minibtch:" + str(learning_rate_minibatch) )
			# 	print("validation_accuracy : " + str(validation_accuracy ))
			# 	predictions = testing(testing_set_normalized,theta_after)
			# 	break
			# else:
			# 	print("no good performace , please try again")

			epochs_minibatch  = 3
			batch_size = 15
			learning_rate_minibatch = 4.5*10**-3
			# learning_rate_minibatch = 0.0006
			(theta_after,cost_training_history, cost_validation_history , training_accuracy, validation_accuracy) = gradient_descent_minibatch(training_feature_normalized, training_yhat,validation_feature_normalized,validation_yhat, theta, learning_rate_minibatch ,epochs_minibatch ,batch_size)

			if( training_accuracy >= training_accuracy_th and validation_accuracy >= validation_accuracy_th):

				print("[normalized] method 0")
				print("sigma:" + str(sigma) + "  learning_rate_minibtch:" + str(learning_rate_minibatch) )
				print("validation_accuracy : " + str(validation_accuracy ))
				(theta_after,cost_training_history, cost_validation_history , training_accuracy, validation_accuracy) = gradient_descent_minibatch(training_feature_total_normalized, training_yhat_total,validation_feature_normalized,validation_yhat, theta, learning_rate_minibatch ,epochs_minibatch ,batch_size)
				predictions = testing(testing_set_normalized,theta_after)
				break
			else:
				print("no good performace , please try again")


			epochs_minibatch  = 3
			batch_size = 10
			learning_rate_minibatch = 4.5*10**-3
			# learning_rate_minibatch = 0.0005
			(theta_after,cost_training_history, cost_validation_history , training_accuracy, validation_accuracy) = gradient_descent_minibatch(training_feature_normalized, training_yhat,validation_feature_normalized,validation_yhat, theta, learning_rate_minibatch ,epochs_minibatch ,batch_size)

			if( training_accuracy >= training_accuracy_th and validation_accuracy >= validation_accuracy_th):

				print("[normalized] method 1")
				print("sigma:" + str(sigma) + "  learning_rate_minibtch:" + str(learning_rate_minibatch) )
				print("validation_accuracy : " + str(validation_accuracy ))
				(theta_after,cost_training_history, cost_validation_history , training_accuracy, validation_accuracy) = gradient_descent_minibatch(training_feature_total_normalized, training_yhat_total,validation_feature_normalized,validation_yhat, theta, learning_rate_minibatch ,epochs_minibatch ,batch_size)
				predictions = testing(testing_set_normalized,theta_after)
				break
			else:
				print("no good performace , please try again")


			epochs_minibatch  = 3
			batch_size = 8
			learning_rate_minibatch = 5.0*10**-3
			# learning_rate_minibatch = 0.0008
			(theta_after,cost_training_history, cost_validation_history , training_accuracy, validation_accuracy) = gradient_descent_minibatch(training_feature_normalized, training_yhat,validation_feature_normalized,validation_yhat, theta, learning_rate_minibatch ,epochs_minibatch ,batch_size)
			if( training_accuracy >= training_accuracy_th and validation_accuracy >= validation_accuracy_th):

				print("[normalized] method 2")
				print("sigma:" + str(sigma) + "  learning_rate_minibtch:" + str(learning_rate_minibatch) )
				print("validation_accuracy : " + str(validation_accuracy ))
				(theta_after,cost_training_history, cost_validation_history , training_accuracy, validation_accuracy) = gradient_descent_minibatch(training_feature_total_normalized, training_yhat_total,validation_feature_normalized,validation_yhat, theta, learning_rate_minibatch ,epochs_minibatch ,batch_size)
				predictions = testing(testing_set_normalized,theta_after)
				break
			else:
				print("no good performace , please try again")





			epochs_minibatch  = 30
			batch_size = 5
			# learning_rate_minibatch = 4.0*10**-3
			learning_rate_minibatch = 0.0008
			(theta_after,cost_training_history, cost_validation_history ,training_accuracy,  validation_accuracy) = gradient_descent_minibatch(training_feature_normalized, training_yhat,validation_feature_normalized,validation_yhat, theta, learning_rate_minibatch ,epochs_minibatch ,batch_size)

			if( training_accuracy >= training_accuracy_th and validation_accuracy >= validation_accuracy_th):

				print("[normalized] method 3")
				print("sigma:" + str(sigma) + "  learning_rate_minibtch:" + str(learning_rate_minibatch) )
				(theta_after,cost_training_history, cost_validation_history , training_accuracy, validation_accuracy) = gradient_descent_minibatch(training_feature_total_normalized, training_yhat_total,validation_feature_normalized,validation_yhat, theta, learning_rate_minibatch ,epochs_minibatch ,batch_size)
				print("validation_accuracy : " + str(validation_accuracy ))
				predictions = testing(testing_set_normalized,theta_after)
				break
			else:
				print("no good performace , please try again")





		if( training_accuracy >= training_accuracy_th and validation_accuracy >= validation_accuracy_th):
			break

	output_filename = "prediction.csv"
	theta_filename = "theta"

	# testing_set = (testing_set)
	
	write_file(predictions, output_filename)

	end_time = time.time()

	min_ = int(  (end_time - start_time)/60  )
	sec_ = (end_time - start_time)%60

	print("Consume : " + str(min_) + "mins " + str(sec_) + "secs" )




	# print("starting point of  theta " + str(theta_tmp))

	'''
	Draw the cost function
	'''
	#gradient descent
	plt.figure(1)
	plot1 = plt.plot(range(len(cost_training_history[0:])) , cost_training_history[0:],'ro' ,label='$training$')
	plot2 = plt.plot(range(len(cost_training_history[0:])) , cost_validation_history[0:],'g--',label ='$Validation$')

	plt.title('Linear Regression')
	plt.title('LR:' + str(learning_rate), loc='left')
	plt.title('epochs :' + str(epochs), loc='right')
	plt.ylabel('Cost')
	plt.xlabel('epochs')
	plt.xlim([0,len(cost_training_history[0:])])
	plt.legend()
	plt.show(block=True)

