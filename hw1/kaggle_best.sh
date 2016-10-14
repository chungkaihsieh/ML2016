#!/usr/bin/env python
import numpy as np 
import pandas as pd 
import scipy
import matplotlib.pyplot as plt 
import csv

'''
read data
'''
# trainning_data = pd.read_csv('../data/train.csv', sep=',' , encoding='latin1' , parse_dates=['Date'], index_col='Date')
columns_select = ['date','location','component']
for i in range(24):
	columns_select.append(str(i))
trainning_data = pd.read_csv('train.csv', sep=',' , encoding='latin1' , names=columns_select)
# trainning_data = pd.read_csv('../data/train.csv', sep=',' , encoding='latin1' , names=['date','location','component','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23'])

columns_select = ['id','component']
for i in range(9):
	columns_select.append(str(i))
testing_data  = pd.read_csv('test_X.csv', sep=',', encoding='latin1', names=columns_select)
# index_testing = testing_data.set_index([[0,1,2,3,4,5,6,7,8,9,10,11]])
# index_testing = testing_data(columns = ['0','1','2','3','4','5','6','7','8','9','10','11'])
# print("testing_data")
# print(testing_data)

'''
data processing
'''

'''
# consequent extract data from days to days
# [0~9],[1~10],...[14~23] : total 15 datas 
#[15~0][16~1]...[23~9]: total:9 datas 
#then cocade together
#[0~9]
# [1~10]
# .
# .
# .
# [14~23]
'''
#-------
# Extract Feature Trainning Data
#-------

#usage extract_feature(pandas dataframe,component string)
def extract_feature(trainning_data,component, feature_num):

	a = []
	a_tmp = []
	frames = []
	columns_select = ['date','component']
	is_component = trainning_data['component'] ==component

	'''
	Turn date string into real "date"
	'''
	# print(trainning_data['date'])
	trainning_data['date'] =  pd.to_datetime(trainning_data['date'][1:])
	# print(trainning_data['date'])
	# print("reading data")

	for month in range(1,13):
		month_extract = ( trainning_data['date'] >= ('2014/' + str(month) + '/1') ) & ( trainning_data['date'] <= ('2014/' + str(month) +'/20') )
		trainning_data_tmp = trainning_data[is_component & month_extract][:]
		# print("trainning_data['date'] :" + str(trainning_data['date']))
		# print("('2014/' + str(month) + '/1') :" + str(('2014/' + str(month) + '/1')))
		# print("('2014/' + str(month+1) +'/1') :" + str(('2014/' + str(month+1) +'/1')))
		# print("month:" + str(month))
		# print("before")
		# print(trainning_data_tmp)
		#reset index
		trainning_data_tmp = trainning_data_tmp.reset_index()
		# print("after")
		# print(trainning_data_tmp)

		#[no] extract feature to next day
		for hour_s in range(0,15):
			for hour_n in range(0,feature_num):
				# print('hour_s=' + str(hour_s), 'hour_n=' + str(hour_n) , 'hour_s+hour_n=' + str( hour_s + hour_n))
				columns_select.append(str( hour_s + hour_n ))
				a.append(str( hour_s + hour_n ))

			tmp = trainning_data_tmp [columns_select] [:]
			# print(tmp)		

			#rename all column  index to 0~9
			for column_index in range(feature_num):
				tmp.rename(columns={a[column_index] : str( column_index )}, inplace=True)
				# print(a[column_index])
				# print(column_index)

			# print(tmp)

			#concatenate  data sets (15 datas/ per day) into one data sets
			frames.append(tmp)
			# print("tmp[0~15]")
			# print(tmp)
			# print("frames")
			# print(frames)
			#reset index
			columns_select = ['date','component']
			a = []

		#extract feature to next day
		for hour_s in range(15,24):
			column_select_part1 = ['date','component']
			column_selcet_part2 = []

			for hour_n in range(0,feature_num):
				# print('hour_s=' + str(hour_s), 'hour_n=' + str(hour_n) , 'hour_s+hour_n=' + str( hour_s + hour_n))
				if( hour_s + hour_n) <24:
					column_select_part1.append( str( hour_s + hour_n )  )
				else:
					column_selcet_part2.append( str( (hour_s + hour_n)%24 )  )

				a.append(str( (hour_s + hour_n)%24 ) )

			tmp_part1 = trainning_data_tmp[column_select_part1][0:19]
			tmp_part2 = trainning_data_tmp[column_selcet_part2][1:20]
			# #reset pandas dataframe index
			tmp_part1 = tmp_part1.reset_index(drop=True)
			tmp_part2 = tmp_part2.reset_index(drop=True)
			tmp = pd.concat([tmp_part1, tmp_part2], axis=1, join_axes=[tmp_part1.index])
			# tmp = pd.concat([tmp_part1, tmp_part2], axis=1, join_axes=[tmp_part1.index], ignore_index=True)		
			# tmp = trainning_data_tmp [columns_select] [:]
			# print("tmp15~24")
			# print(tmp)
			# print("tmp_part1")
			# print(tmp_part1)
			# print("tmp_part2")
			# print(tmp_part2)		
			# print("tmp")
			# print(tmp)

			'''
			Bugs for columns alignment
			'''

			#rename all column  index to 0~9
			for column_index in range(feature_num):
				tmp.rename(columns={a[column_index] : str( column_index+100 )}, inplace=True)
				# print("original index:" + str(a[column_index]) )
				# print("new index:" + str(column_index) )
			for column_index in range(feature_num):
				tmp.rename(columns={str(column_index + 100) : str( column_index )}, inplace=True)
				# print("original index:" + str(column_index + 100))
				# print("new index:" + str(column_index) )

			# print("tmp15~24")
			# print(tmp)

			#concatenate  data sets (15 datas/ per day) into one data sets
			frames.append(tmp)
			#reset index
			columns_select = ['date','component']
			a = []	
			# print("frames")
			# print(frames)
	#trainning_data_15dpd = trainning data (15 Datas per Day)
	trainning_data_24dpd = pd.concat(frames)
	return trainning_data_24dpd
	# print(frames)

# trainning_data_24dpd = extract_feature(trainning_data,"PM2.5",10)
# component_extract_list = ["PM2.5","AMB_TEMP","CH4","CO","NMHC","NO","NO2","NOx","O3","PM10","RH","SO2","THC","WD_HR","WIND_DIREC","WIND_SPEED","WS_HR"]
component_extract_list = ["PM2.5","CO","NO2","O3","PM10","SO2"]
# component_extract_list = ["PM2.5","PM10"]

# component_extract_list = ["PM2.5","AMB_TEMP","CH4"]

'''
Trainning Data
'''
# print("component length:" + str(len(component_extract_list)))
trainning_data_24dpd_tmp = [None]*len(component_extract_list)

for i in range(len(component_extract_list)):
	print("extract component:" + str(component_extract_list[i]))
	#the 10's element is label
	if( i == 0 ):
		trainning_data_24dpd_tmp[i] = extract_feature(trainning_data, component_extract_list[i], 10)
	else:
		trainning_data_24dpd_tmp[i] = extract_feature(trainning_data, component_extract_list[i], 9)

# print(trainning_data_24dpd_tmp[0])
# print(trainning_data_24dpd_tmp[1])

'''
Testing Data
'''
# #-------
# # Extract Feature testing Data
# #-------
# is_PM25 = testing_data['component'] =="PM2.5"
# testing_set_tmp = testing_data[is_PM25][[2,3,4,5,6,7,8,9,10]]
# new_index = list(range( len(testing_set_tmp.index) )) 


# testing_set = testing_set_tmp.as_matrix(columns=testing_data.columns[2:]).astype(dtype='float32')
# # print(testing_set_tmp)
# print(testing_set)

# is_component = trainning_data['component'] ==component

testing_data_tmp = [None]*len(component_extract_list)

for i in range(len(component_extract_list)):
	# print("extract component:" + str(component_extract_list[i]))
	#the 10's element is label
	is_component = testing_data['component'] == component_extract_list[i]
	testing_data_tmp[i] = testing_data[is_component][['0','1','2','3','4','5','6','7','8']]






#-----------
#turn pandas: DataFrame ---> Numpy.array()
#-----------
'''
Trainning Set - feature_set
'''


columns_select = ['0','1','2','3','4','5','6','7','8']
# trainning_set_tmp = trainning_data_15dpd.as_matrix(columns= trainning_data_15dpd.columns[1:] ).astype(dtype = 'float32')
# trainning_set_tmp = trainning_data_24dpd.as_matrix(columns= columns_select).astype(dtype = 'float32')
trainning_set_tmp = [None]*len(component_extract_list)

# print("trainning_set_tmp")
# print(trainning_set_tmp)
for i in range(len( component_extract_list )):
	trainning_set_tmp[i] = trainning_data_24dpd_tmp[i].as_matrix(columns= columns_select).astype(dtype = 'float32')
	if i ==0 :
		trainning_feature = trainning_set_tmp[i]
	else:
		trainning_feature = np.concatenate([trainning_feature, trainning_set_tmp[i] ], axis=1)
'''
trainning Set - yhat
'''
columns_select = ['9']
trainning_yhat = trainning_data_24dpd_tmp[0].as_matrix(columns= columns_select).astype(dtype = 'float32')

# print("yhat_set")
# print(trainning_yhat)
# print(trainning_yhat.shape)

'''
Testing Set
'''

columns_select = ['0','1','2','3','4','5','6','7','8']
# trainning_set_tmp = trainning_data_15dpd.as_matrix(columns= trainning_data_15dpd.columns[1:] ).astype(dtype = 'float32')
# trainning_set_tmp = trainning_data_24dpd.as_matrix(columns= columns_select).astype(dtype = 'float32')
testing_set_tmp = [None]*len(component_extract_list)

# print("trainning_set_tmp")
# print(trainning_set_tmp)
for i in range(len( component_extract_list )):
	testing_set_tmp[i] = testing_data_tmp[i].as_matrix(columns= columns_select).astype(dtype = 'float32')
	if i ==0 :
		testing_set = testing_set_tmp[i]
	else:
		testing_set = np.concatenate([testing_set, testing_set_tmp[i] ], axis=1)


#-------
# Turn data into:
# [1] trainning_set
# [2] testing_set
# [3] validation_set
#-------

#Turn Trainning set -> (1)trainning_set, (2)validation_set
validation_percent = 1/8.0
trainning_percent = 1 - validation_percent
num_trainning = int(trainning_percent * trainning_feature.shape[0])

indices = np.random.permutation(trainning_feature.shape[0])
trainning_idx, validation_idx = indices[: num_trainning], indices[num_trainning:]
# trainning_set,validation_set = trainning_set_tmp[trainning_idx,:] , trainning_set_tmp[validation_idx,:] 
trainning_feature,validation_feature = trainning_feature[trainning_idx,:] , trainning_feature[validation_idx,:]
trainning_yhat,validation_yhat = trainning_yhat[trainning_idx,:] , trainning_yhat[validation_idx,:]






# '''------------------------------------------
# [Algorithm] - linear regression
# ------------------------------------------'''


#usage: 
#data_set=numpy.array([])
def feature_normalize(data_set):
	tmp = data_set
	for i in range(np.shape(tmp)[1]):
		tmp[: , i] = ( tmp[:,i] - np.mean(tmp[:,i])) / np.std(tmp[:,i])

	return tmp

def rms(array,length):
	return np.sqrt( (1.0/length) * (array.T.dot(array)) )
def rss(array):
	return np.sqrt(array.T.dot(array))

#usage:
# X = numpy.array([]) = np.array([x1,x2...xn])
# X_tmp = numpy.array([]) = np.array([x1,x2...xn, 1 ])
# y = numpy.array([]) = np.array([y1,y2...yn])
# theta = numpy.array([]) = np.array([w1,w2...wn, b ])
def compute_cost(X, y, theta):
	#----------------------------------------------
	# Compute cost  for linear regression
	#----------------------------------------------
	X_tmp = X
	n = X.shape[0]
	y.shape = (n,1)
	bias_ones = np.ones( (n,1) , dtype='float32')
	X_tmp = np.append(X, bias_ones, axis=1)

	predictions = X_tmp.dot(theta)
	predictions.shape = (n,1)
	sqErrors = (y - predictions)

	cost = (1.0/(2 * n)) * sqErrors.T.dot(sqErrors)
	return cost

def compute_cost_regularization(X, y, theta, regular_coeff):
	#----------------------------------------------
	# Compute cost  for linear regression
	#----------------------------------------------
	X_tmp = X
	n = X.shape[0]
	y.shape = (n,1)
	bias_ones = np.ones( (n,1) , dtype='float32')
	X_tmp = np.append(X, bias_ones, axis=1)

	predictions = X_tmp.dot(theta)
	predictions.shape = (n,1)
	sqErrors = (y - predictions)

	cost = (1.0/(2 * n)) * sqErrors.T.dot(sqErrors) + regular_coeff*(theta.T.dot(theta))
	return cost



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
	X_validation_tmp = np.append(X, bias_ones, axis=1)

	g_history = np.zeros( [len(theta), epochs] )
	cost_trainning_history = np.zeros(shape=(epochs, 1))
	cost_validation_history = np.zeros(shape=(epochs, 1))

	for i in range(epochs):
		predictions = X_tmp.dot(theta)
		theta_size = theta.shape[0]



		for it in range(theta_size):
			temp = X_tmp[:,it]
			temp.shape = (n,1)

			error_x1 = ( y - predictions) * (-temp)
			g = (1.0/n) * error_x1.sum()
			g_history[it, i] = g
			# print("g_histroy : " + str(g_history))
			# print("rms          : " + str(rms( g_history[it,:], i+1)))
			adagrad_coeff = rss( g_history[it,:])
			# error_x1 = ( y - predictions) * (temp)
			# theta[it] = theta[it] - learning_rate * g
			theta[it] = theta[it] - learning_rate * g/adagrad_coeff			



		print("iteration numbers : " + str(i))
		# print("theta : " + str(theta))
		print(" [cost] - trainning data   : " + str( compute_cost(X, y, theta)))
		print(" [cost] - validation data : " + str( compute_cost(X_validation, y_validation, theta)))
		cost_trainning_history[i, 0] = compute_cost(X, y, theta)
		cost_validation_history[i,0] = compute_cost(X_validation,y_validation,theta)

	return theta, cost_trainning_history, cost_validation_history




#***************************************
#-----Gradient (adagrad + minibatch)-----
#***************************************

def gradient_descent_minibatch(X, y, X_validation, y_validation, theta, learning_rate, epochs, batch_size):
	#-------------------------------------------------
	# Performs gradient descent to learn theta
	# by taking epochs gradient steps with learning_rate
	#-------------------------------------------------
	for i in range(epochs):
		for batch_num in range(int(X.shape[0]/batch_size)):

			X_tmp = X[batch_num * batch_size:batch_num * batch_size + batch_size,:]
			y_tmp = y[batch_num * batch_size:batch_num * batch_size + batch_size,:]
			n = X_tmp.shape[0]
			bias_ones = np.ones( (n,1) , dtype='float32')
			X_tmp = np.append(X_tmp, bias_ones, axis=1)

			n_tmp = X.shape[0]
			bias_ones_tmp = np.ones( (n_tmp,1) , dtype='float32')
			X_validation_tmp = X_validation
			X_validation_tmp = np.append(X, bias_ones_tmp, axis=1)

			g_history = np.zeros( [len(theta), epochs] )
			cost_trainning_history = np.zeros(shape=(epochs, 1))
			cost_validation_history = np.zeros(shape=(epochs, 1))

			predictions = X_tmp.dot(theta)
			theta_size = theta.shape[0]

			for it in range(theta_size):
				temp = X_tmp[:,it]
				temp.shape = (n,1)

				if(i <=500):
					error_x1 = ( y_tmp - predictions) * (-temp)
					g = (1.0/n) * error_x1.sum()
					g_history[it, i] = g
					# print("g_histroy : " + str(g_history))
					# print("rms          : " + str(rms( g_history[it,:], i+1)))
					adagrad_coeff = rss( g_history[it,:])
					# error_x1 = ( y - predictions) * (temp)
					theta[it] = theta[it] - learning_rate * g/adagrad_coeff
				else:
					learning_rate = 0.000001
					error_x1 = ( y_tmp - predictions) * (-temp)
					g = (1.0/n) * error_x1.sum()
					g_history[it, i] = g
					# print("g_histroy : " + str(g_history))
					# print("rms          : " + str(rms( g_history[it,:], i+1)))
					adagrad_coeff = rss( g_history[it,:])
					# error_x1 = ( y - predictions) * (temp)
					theta[it] = theta[it] - learning_rate * g
		print("iteration numbers : " + str(i))
		# print("theta : " + str(theta))
		print(" [cost minibatch] - trainning data   : " + str( compute_cost(X, y, theta)))
		print(" [cost minibatch] - validation data : " + str( compute_cost(X_validation, y_validation, theta)))
		cost_trainning_history[i, 0] = compute_cost(X, y, theta)
		cost_validation_history[i,0] = compute_cost(X_validation,y_validation,theta)

	return theta, cost_trainning_history, cost_validation_history



#********************************************
#-----Gradient (adagrad + momentum)-----
#********************************************

def gradient_descent_momentum(X, y, X_validation, y_validation, theta, learning_rate, epochs, movement_parameter):
	#-------------------------------------------------
	# Performs gradient descent to learn theta
	# by taking epochs gradient steps with learning_rate
	#-------------------------------------------------
	X_tmp = X
	n = X.shape[0]
	bias_ones = np.ones( (n,1) , dtype='float32')
	X_tmp = np.append(X, bias_ones, axis=1)

	X_validation_tmp = X_validation
	X_validation_tmp = np.append(X, bias_ones, axis=1)

	g_history = np.zeros( [len(theta), epochs] )
	cost_trainning_history = np.zeros(shape=(epochs, 1))
	cost_validation_history = np.zeros(shape=(epochs, 1))

	'''
	momentum
	'''
	movement = np.zeros( shape = (len(theta),1))

	for i in range(epochs):
		predictions = X_tmp.dot(theta)
		theta_size = theta.shape[0]

		for it in range(theta_size):
			temp = X_tmp[:,it]
			temp.shape = (n,1)

			error_x1 = ( y - predictions) * (-temp)
			g = (1.0/n) * error_x1.sum()
			g_history[it, i] = g
			# print("g_histroy : " + str(g_history))
			# print("rms          : " + str(rms( g_history[it,:], i+1)))
			adagrad_coeff = rss( g_history[it,:])
			# error_x1 = ( y - predictions) * (temp)
			'''
			momentum
			'''
			movement[it] = movement_parameter * movement[it] - learning_rate * g/adagrad_coeff
			theta[it] = theta[it] + movement[it]

		print("iteration numbers : " + str(i))
		# print("theta : " + str(theta))
		print(" [cost momentum] - trainning data   : " + str( compute_cost(X, y, theta)))
		print(" [cost momentum] - validation data : " + str( compute_cost(X_validation, y_validation, theta)))
		cost_trainning_history[i, 0] = compute_cost(X, y, theta)
		cost_validation_history[i,0] = compute_cost(X_validation,y_validation,theta)

	return theta, cost_trainning_history, cost_validation_history



#********************************************
#-----Gradient (adagrad + regularization)-----
#********************************************

def gradient_descent_regularization(X, y, X_validation, y_validation, theta, learning_rate, epochs, regular_coeff):
	#-------------------------------------------------
	# Performs gradient descent to learn theta
	# by taking epochs gradient steps with learning_rate
	#-------------------------------------------------
	X_tmp = X
	n = X.shape[0]
	bias_ones = np.ones( (n,1) , dtype='float32')
	X_tmp = np.append(X, bias_ones, axis=1)

	X_validation_tmp = X_validation
	X_validation_tmp = np.append(X, bias_ones, axis=1)


	# cost_history = np.zeros(shape=(epochs, 1))
	g_history = np.zeros( [len(theta), epochs] )
	cost_trainning_history = np.zeros(shape=(epochs, 1))
	cost_validation_history = np.zeros(shape=(epochs, 1))

	for i in range(epochs):
		predictions = X_tmp.dot(theta)
		theta_size = theta.shape[0]

		for it in range(theta_size):
			temp = X_tmp[:,it]
			temp.shape = (n,1)

			error_x1 = ( y - predictions) * (-temp)
			g = (1.0/n) * error_x1.sum()
			g_history[it, i] = g

			adagrad_coeff = rss( g_history[it,:])
			regularization_component = 2*regular_coeff*theta.sum()
			# error_x1 = ( y - predictions) * (temp)

			theta[it] = theta[it] - learning_rate * ( g/adagrad_coeff + regularization_component )
		print("iteration numbers : " + str(i))
		# print("theta : " + str(theta))
		print(" [cost regularization] - trainning data   : " + str( compute_cost_regularization(X, y, theta, regular_coeff)))
		print(" [cost regularization] - validation data : " + str( compute_cost_regularization(X_validation, y_validation, theta, regular_coeff)))
		cost_trainning_history[i, 0] = compute_cost(X, y, theta)
		cost_validation_history[i,0] = compute_cost(X_validation,y_validation,theta)

	return theta, cost_trainning_history, cost_validation_history

#usage:
# X = numpy.array([]) = np.array([x1,x2...xn])
# X_tmp = numpy.array([]) = np.array([x1,x2...xn, 1 ])
# y = numpy.array([]) = np.array([y1,y2...yn])
# theta = numpy.array([]) = np.array([w1,w2...wn, b ])
#learning_rate = float
#epochs = int

def testing(X,theta):
	X_tmp = X
	n = X.shape[0]
	bias_ones = np.ones( (n,1) , dtype='float32')
	X_tmp = np.append(X, bias_ones, axis=1)
	predictions_tmp = X_tmp.dot(theta)
	predictions = np.rint( predictions_tmp ).astype(int)
	return  predictions



def write_file(predictions,file_name):
	predictions.shape = (len(predictions) , 1)
	writer = open(file_name,"w")
	writer.write("id,value\n")
	for i in range(len(predictions)):
		str_tmp = "id_"+str(i)+"," + str(predictions[i][0]) + "\n"
		# print(str_tmp)
		writer.write(str_tmp)		
	writer.close()



if __name__ == '__main__':
	'''
	Read Data
	'''
	# print("trainning_set shape:" + str(trainning_set[:,0:10].shape))
	# trainning_feature =  trainning_set[:,0:9]
	# trainning_yhat = trainning_set[:,9:10]
	# validation_feature = validation_set[:,0:9]
	# validation_yhat = validation_set[:,9:10]

	# print("trainning_set shape:" + str(trainning_set[:,0:10].shape))
	# trainning_feature =   trainning_set[:,0:9] 
	# trainning_yhat = trainning_set[:,9:10]
	# validation_feature =  validation_set[:,0:9] 
	# validation_yhat = validation_set[:,9:10]


	'''
	Test for all method
	'''

	theta_count = 1000
	output_filename = "kaggle_best.csv"
	theta_filename = "theta"


	mu, sigma = 0, 0.8
	theta = sigma * np.random.randn(1, 9*len(component_extract_list) +1).T + mu
	# theta = np.zeros([1,10]).T
	theta_tmp = theta
	# theta = (np.random.normal(mu, sigma,10)).T
	# theta = (np.random.rand(1,10).T - 0.5)*2
	epochs = 3*(10**4)
	learning_rate = 0.25
	learning_rate_minibatch = 0.25
	regularization_parameter = 0.001
	momentum_parameter = 0.05
	batch_size = 5
	# print(trainning_set)
	# print(trainning_feature)
	# print(trainning_yhat)
	# gradient_descent(trainning_feature, trainning_yhat, theta, 0.0001,1500)

	# trainning_feature = feature_normalize(trainning_feature)
	# validation_feature = feature_normalize(validation_feature)
	# testing_set = feature_normalize(testing_set)

	(theta,cost_trainning_history, cost_validation_history) = gradient_descent(trainning_feature, trainning_yhat,validation_feature,validation_yhat, theta, learning_rate ,epochs)
	print("**************************************")
	print("****  parameter count :" + str(i) + "******")
	print("**************************************")

	# output_filename = output_filename_tmp + str(i) + ".csv"
	# theta_filename = theta_filename_tmp + str(i)
	predictions = testing(testing_set,theta)
	write_file(predictions, output_filename)
	# np.save(theta_filename, theta_tmp)

	# (theta_b,cost_trainning_history_b, cost_validation_history_b) = gradient_descent_minibatch(trainning_feature, trainning_yhat,validation_feature,validation_yhat, theta, learning_rate_minibatch ,epochs,batch_size)
	# predictions = testing(testing_set,theta_b)
	# write_file(predictions,"linear_regression_minibatch.csv")
	# (theta_r,cost_trainning_history_r, cost_validation_history_r) = gradient_descent_regularization(trainning_feature, trainning_yhat,validation_feature,validation_yhat, theta, learning_rate ,epochs,regularization_parameter)
	# (theta_m,cost_trainning_history_m, cost_validation_history_m) = gradient_descent_momentum(trainning_feature, trainning_yhat,validation_feature,validation_yhat, theta, learning_rate ,epochs,momentum_parameter)

	# (theta_regular,cost_regular) = gradient_descent_regularization(trainning_feature, trainning_yhat,validation_feature,validation_yhat, theta, learning_rate,epochs, 0.01)




	# predictions = testing(testing_set,theta_r)
	# write_file(predictions,"linear_regression_regularization.csv")

	# predictions = testing(testing_set,theta_m)
	# write_file(predictions,"linear_regression_momentum.csv")



	print("starting point of  theta " + str(theta_tmp))

	'''
	Draw the cost function
	'''
	#gradient descent
	plt.figure(1)
	plot1 = plt.plot(range(len(cost_trainning_history[100:])) , cost_trainning_history[100:],'ro' ,label='$Trainning$')
	plot2 = plt.plot(range(len(cost_trainning_history[100:])) , cost_validation_history[100:],'g--',label ='$Validation$')

	plt.title('Linear Regression')
	plt.title('LR:' + str(learning_rate), loc='left')
	plt.title('epochs :' + str(epochs), loc='right')
	plt.ylabel('Cost')
	plt.xlabel('epochs')
	plt.xlim([100,len(cost_trainning_history[100:])])
	plt.legend()
	plt.show(block=True)

	# first_legend = plt.legend( handles=[plot1], loc=1)
	# plt.legend( [plot1,plot2] , ('Trainning','Validation'), 'best' , numpoints=1)

	# #regularization
	# plt.figure(2)
	# plot1 = plt.plot(range(len(cost_trainning_history_r[100:])) , cost_trainning_history_r[100:],'ro' ,label='$Trainning_r$')
	# plot2 = plt.plot(range(len(cost_trainning_history_r[100:])) , cost_validation_history_r[100:],'g--',label ='$Validation_r$')

	# plt.title('Linear Regression[regularization]')
	# plt.title('LR:' + str(learning_rate), loc='left')
	# # plt.title('epochs :' + str(epochs), loc='right')
	# plt.ylabel('Cost')
	# plt.xlabel('epochs')
	# plt.xlim([100,len(cost_trainning_history_r[100:])])
	# plt.legend()
	# # first_legend = plt.legend( handles=[plot1], loc=1)
	# # plt.legend( [plot1,plot2] , ('Trainning','Validation'), 'best' , numpoints=1)

	# #momentum
	# plt.figure(3)
	# plot1 = plt.plot(range(len(cost_trainning_history_m[100:])) , cost_trainning_history_m[100:],'ro' ,label='$Trainning_r$')
	# plot2 = plt.plot(range(len(cost_trainning_history_m[100:])) , cost_validation_history_m[100:],'g--',label ='$Validation_r$')

	# plt.title('Linear Regression[momentum]')
	# plt.title('LR:' + str(learning_rate), loc='left')
	# # plt.title('epochs :' + str(epochs), loc='right')
	# plt.ylabel('Cost')
	# plt.xlabel('epochs')
	# plt.xlim([100,len(cost_trainning_history_m[100:])])
	# plt.legend()
	# # first_legend = plt.legend( handles=[plot1], loc=1)
	# # plt.legend( [plot1,plot2] , ('Trainning','Validation'), 'best' , numpoints=1)

	#minibatch
	# plt.figure(4)
	# plot1 = plt.plot(range(len(cost_trainning_history_b[100:])) , cost_trainning_history_b[100:],'ro' ,label='$Trainning$')
	# plot2 = plt.plot(range(len(cost_trainning_history_b[100:])) , cost_validation_history_b[100:],'g--',label ='$Validation$')

	# plt.title('Linear Regression')
	# plt.title('LR:' + str(learning_rate), loc='left')
	# plt.title('epochs :' + str(epochs), loc='right')
	# plt.ylabel('Cost')
	# plt.xlabel('epochs')
	# plt.xlim([100,len(cost_trainning_history[100:])])
	# plt.legend()


	# first_legend = plt.legend( handles=[plot1], loc=1)
	# plt.legend( [plot1,plot2] , ('Trainning','Validation'), 'best' , numpoints=1)


	# '''
	# Testing for learnning rate
	# '''
	# mu, sigma = 0, 1.5
	# theta = sigma * np.random.randn(1,10).T + mu
	# theta_tmp = theta
	# # theta = (np.random.normal(mu, sigma,10)).T

	# # theta = (np.random.rand(1,10).T - 0.5)*2

	# epochs = 1000
	# regularization_parameter = 0.001
	# momentum_parameter = 0.01
	# learning_rate_tmp = 0.01
	# plt.figure(1)

	# for i in range (1,9):
	# 	learning_rate  = (0.05*i) + learning_rate_tmp  
	# 	(theta,cost_trainning_history, cost_validation_history) = gradient_descent(trainning_feature, trainning_yhat,validation_feature,validation_yhat, theta, learning_rate ,epochs)

	# 	color = ['r','g','b','c','m','y','k','w']
	# 	'''
	# 	Draw the cost function
	# 	'''
	# 	index_show = 100

	# 	plot1 = plt.plot(range(len(cost_trainning_history[index_show:])) , cost_trainning_history[index_show:],  color[i-1]+'o' , label=('$'+"T_LR:"+str(learning_rate)+'$' ))
	# 	plot2 = plt.plot(range(len(cost_trainning_history[index_show:])) , cost_validation_history[index_show:], color[i-1]+'--', label =('$'+"V_LR:"+str(learning_rate)+'$'))

	# 	plt.title('Linear Regression')
	# 	# plt.title('LR:' + str(0.5), loc='left')
	# 	plt.title('epochs :' + str(epochs), loc='right')
	# 	plt.ylabel('Cost')
	# 	plt.xlabel('epochs')
	# 	plt.xlim([index_show,len(cost_trainning_history[index_show:])])
	# 	plt.legend()
	# 	# first_legend = plt.legend( handles=[plot1], loc=1)
	# 	# plt.legend( [plot1,plot2] , ('Trainning','Validation'), 'best' , numpoints=1)


	# plt.show(block=True)
