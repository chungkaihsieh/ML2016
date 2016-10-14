#!/usr/bin/env python
import numpy as np 
import pandas as pd 
import scipy 
import csv

'''
plot the graph
'''
pd.set_option('display.mpl_style' , 'default')
pd.set_option('display.line_width', 5000) 
pd.set_option('display.max_columns', 60) 
# figsize(15, 5)

'''
read data
'''
trainning_data = pd.read_csv('../data/train.csv', sep=',' , encoding='latin1' , parse_dates=['Date'], index_col='Date')
trainning_data = pd.read_csv('../data/train.csv', sep=',' , encoding='latin1' , names=['date','location','component','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23'])
testing_data  = pd.read_csv('../data/test_X.csv', sep=',', encoding='latin1', names=['id' , 'component','0','1','2','3','4','5','6','7','8','9',])
# index_testing = testing_data.set_index([[0,1,2,3,4,5,6,7,8,9,10,11]])
# index_testing = testing_data(columns = ['0','1','2','3','4','5','6','7','8','9','10','11'])
print(testing_data)

'''
data processing
'''

'''
# consequent extract data from days to days
# [0~9],[1~10],...[14~23] : total 15 datas per days
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

# print(trainning_data[:20])
#list component column
# print(trainning_data[['location','component','0','5']])
#list all PM2.5 component
# print(trainning_data[trainning_data.component == "PM2.5"])
#[high class]  list PM2.5 component 
is_PM25 = trainning_data['component'] =="PM2.5"

# print(trainning_data[is_PM25] ['20140101':'20140220'])
# print(trainning_data[is_PM25 ] [['component', '0','1','2' ]] ['20140101':'20140220'])

a = []
frames = []

for i in range(15):
	#extract data [0~9],[1~10]...[14~23]
	for j in range(10):
		print('i=' + str(i), 'j=' + str(j) , 'i+j=' + str(i+j))
		a.append(str( i+j ))
	# tmp = trainning_data[is_PM25 ] [['component', a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9] ]] ['20140101':'20140102']
	tmp = trainning_data[is_PM25 ] [['component', a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9] ]] [:]
	#rename all column  index to 0~9
	for column_index in range(10):
		tmp.rename(columns={a[column_index] : str( int(a[column_index]) - i )}, inplace=True)
		print(a[column_index])
		print(str( int(a[column_index]) - i ))

	print(tmp)

	#concatenate  data sets (15 datas/ per day) into one data sets
	frames.append(tmp)
	#reset index
	a = []
#trainning_data_15dpd = trainning data (15 Datas per Day)
trainning_data_15dpd = pd.concat(frames)



# # print(trainning_data[is_PM25])
# # print(traininng_data["date"])
# # print( trainning_data.head())
# print(trainning_data["date"][1])
# print(trainning_data["date"][1] =="2014/1/1")
# # print(trainning_data[is_PM25 ] [['component', '0','1','2' ]] ['20140101':'20140120'])


# a = []
# frames = []
# frames_tmp = []

# print(trainning_data[is_PM25 ] [['date','component', '0','1','2','3','4' ]].reset_index() )

# # print( pd.concat([trainning_data[is_PM25 ] [['component', a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9] ]] [1],trainning_data[is_PM25 ] [['component', a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9] ]] [1]], axis=1, ignore_indext=True ))

# trainning_data_tmp = trainning_data[is_PM25] [['date','component', '0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']].reset_index()
# print(trainning_data_tmp[1:2])
# print(trainning_data_tmp[1:2] [['date','component', '0','1','2','3','4','5','6','7','8','9']])
# print(trainning_data_tmp[1:2] [['date','component', '0','1','2','3','4','5','6','7','8','9']].append(trainning_data_tmp[1:2] [['date','component', '0','1','2','3','4','5','6','7','8','9']]))
# print(pd.concat([trainning_data_tmp[1:2] [['date','component', '0','1','2','3','4','5','6','7','8','9']],trainning_data_tmp[1:2] [['0','1','2','3','4','5','6','7','8','9']]],axis=1))
# # a = trainning_data_tmp[['component','0','1','2','3','4','5','6','7','8','9']] [1]
# # b = pd.concat([a,a])
# # print(b)
# # print(trainning_data_tmp["date"])
# # print(trainning_data_tmp["date"][1])


# for k in range(len(trainning_data_tmp.index)):
# 	if(trainning_data_tmp["date"][k] == "2014/1/20" or trainning_data_tmp["date"][k] == "2014/2/20" or trainning_data_tmp["date"][k] == "2014/3/20" or trainning_data_tmp["date"][k] == "2014/4/20" or trainning_data_tmp["date"][k] == "2014/5/20" or trainning_data_tmp["date"][k] == "2014/6/20" or trainning_data_tmp["date"][k] == "2014/7/20" or trainning_data_tmp["date"][k] == "2014/8/20" or trainning_data_tmp["date"][k] == "2014/9/20" or trainning_data_tmp["date"][k] == "2014/10/20" or trainning_data["date"][k] == "2014/11/20" or trainning_data["date"][k] == "2014/12/20"):
# 		for i in range(15):
# 			for j in range(10):
# 				print('i=' + str(i), 'j=' + str(j) , 'i+j=' + str(i+j))
# 				a.append(str( i+j ))
# 			# print("a :" + str(a))
# 			# print("k:" + str(k))
# 			# for l in range(10):
# 			# 	print(a[l])
# 			# print( trainning_data_tmp[k:k+1])
# 			tmp = trainning_data_tmp[k:k+1] [['component', a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9] ]]
# 			print("tmp type:" + str(type(tmp)))
# 			# tmp = trainning_data_tmp[1:2] [['date','component', str(i+j+0),str(i+j+0),str(i+j+0),str(i+j+0),str(i+j+0),str(i+j+0),str(i+j+0),str(i+j+0),str(i+j+0),str(i+j+0)]]
# 			# tmp = trainning_data_tmp[1:2] [['date','component', '0','1','2','3','4','5','6','7','8','9']]
# 			for column_index in range(10):
# 				tmp.rename(columns={a[column_index] : str( int(a[column_index]) - i )}, inplace=True)

# 			frames.append(tmp)
# 			a = []
# 	else:
# 		for i in range(24):
# 			for j in range(10):

# 				# print('i=' + str(i), 'j=' + str(j) , 'i+j=' + str(i+j))
# 				index = (i + j) % 24
# 				next_day = ( i+j)/24
# 				a.append(str(index))
# 			# print("a" + str(a))
# 			if(next_day == 0 ):
# 				# print("a" + str(a))
# 				# print("k:" + str(k))
# 				# print( trainning_data_tmp[k:k+1])
# 				# print("error")
# 				tmp = trainning_data_tmp[k:k+1] [['component', a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9] ]]					
# 			else:
# 				# print("a" + str(a))
					
# 				if (a[0] == 15):
# 					tmp = pd.concat([trainning_data_tmp[k:k+1] [['component', a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8] ]],trainning_data_tmp[k+1:k+2] [[a[9] ]]],axis=1)
# 				elif(a[0] == 16):
# 					tmp = pd.concat([trainning_data_tmp[k:k+1] [['component', a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7] ]],trainning_data_tmp[k+1:k+2] [[a[8],a[9] ]]],axis=1)
# 				elif(a[0] == 17):
# 					tmp = pd.concat([trainning_data_tmp[k:k+1] [['component', a[0],a[1],a[2],a[3],a[4],a[5],a[6] ]],trainning_data_tmp[k+1:k+2] [[a[7],a[8],a[9] ]]],axis=1)
# 				elif(a[0] == 18):
# 					tmp = pd.concat([trainning_data_tmp[k:k+1] [['component', a[0],a[1],a[2],a[3],a[4],a[5] ]],trainning_data_tmp[k+1:k+2] [[a[6],a[7],a[8],a[9] ]]],axis=1)
# 				elif(a[0] == 19):
# 					tmp = pd.concat([trainning_data_tmp[k:k+1] [['component', a[0],a[1],a[2],a[3],a[4] ]],trainning_data_tmp[k+1:k+2] [[a[5],a[6],a[7],a[8],a[9] ]]],axis=1)
# 				elif(a[0] == 20):
# 					tmp = pd.concat([trainning_data_tmp[k:k+1] [['component', a[0],a[1],a[2],a[3] ]],trainning_data_tmp[k+1:k+2] [[a[4],a[5],a[6],a[7],a[8],a[9] ]]],axis=1)
# 				elif(a[0] == 21):
# 					tmp = pd.concat([trainning_data_tmp[k:k+1] [['component', a[0],a[1],a[2] ]],trainning_data_tmp[k+1:k+2] [[a[3],a[4],a[5],a[6],a[7],a[8],a[9] ]]],axis=1)
# 				elif(a[0] == 22):
# 					tmp = pd.concat([trainning_data_tmp[k:k+1] [['component', a[0],a[1] ]],trainning_data_tmp[k+1:k+2] [[a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9] ]]],axis=1)
# 				else:
# 					tmp = pd.concat([trainning_data_tmp[k:k+1] [['component', a[0] ]],trainning_data_tmp[k+1:k+2] [[a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9] ]]],axis=1)

# 			for column_index in range(10):
# 				tmp.rename(columns={a[column_index] : str( int(a[column_index]) - i )}, inplace=True)


# 			print(tmp)
# 			frames.append(tmp)
# 			a = []

# trainning_data_24dpd = pd.concat(frames)
# print(trainning_data_24dpd)
# #--------------------------------------------------------------------------------------------------------------------

# print(trainning_data_15dpd)
# print(trainning_data_15dpd[0:5])
# print(trainning_data_15dpd.iloc[0:6,1:10])
#-----------
#turn pandas: DataFrame ---> Numpy.array()
#-----------
trainning_set_tmp = trainning_data_15dpd.as_matrix(columns= trainning_data_15dpd.columns[1:] ).astype(dtype = 'float32')
# trainning_set_tmp = trainning_data_24dpd.as_matrix(columns= trainning_data_24dpd.columns[1:] ).astype(dtype = 'float32')
# print("trainning_set shape:" + str(trainning_set_tmp.shape))
# print(trainning_set_tmp)
# print(type( trainning_set_tmp ))
# print(trainning_set_tmp[1,1])

#-------
# Extract Feature testing Data
#-------
is_PM25 = testing_data['component'] =="PM2.5"
testing_set_tmp = testing_data[is_PM25][[2,3,4,5,6,7,8,9,10]]
new_index = list(range( len(testing_set_tmp.index) )) 

# print("testing_data columns" + str(testing_data.columns))
# print("testing_data testing_data.columns[0]" + str(testing_data.columns[0]))
# print("testing_data testing_data.columns[1]" + str(testing_data.columns[1]))
# print("testing_data testing_data.columns[5]" + str(testing_data.columns[5]))

testing_set = testing_set_tmp.as_matrix(columns=testing_data.columns[2:]).astype(dtype='float32')
# print(testing_set_tmp)
print(testing_set)


#-------
# Turn data into:
# [1] trainning_set
# [2] testing_set
# [3] validation_set
#-------

#Turn Trainning set -> (1)trainning_set, (2)validation_set
validation_percent = 0.1
trainning_percent = 1 - validation_percent
num_trainning = int(trainning_percent * trainning_set_tmp.shape[0])

indices = np.random.permutation(trainning_set_tmp.shape[0])
trainning_idx, validation_idx = indices[: num_trainning], indices[num_trainning:]
trainning_set,validation_set = trainning_set_tmp[trainning_idx,:] , trainning_set_tmp[validation_idx,:] 
# print("trainning set")
# print(trainning_set)
# print(trainning_set[1,:])
# print(trainning_set[:,1])
# print("trainning set shape         :" + str(trainning_set.shape))
# print("Total number of data      :" + str(trainning_set_tmp.shape[0]))
# print("number of rainning set   :" + str(trainning_set.shape[0]))
# print("number of validation set:" + str(validation_set.shape[0]))




# '''
# For Testing
# '''
# #-----------------------------------------------------------------------------------------------------------------------------


# # print(type(trainning_data_15dpd.iloc[0,5]))
# # print(int(trainning_data_15dpd.iloc[0,5]))
# # print(type(int(trainning_data_15dpd.iloc[0,5])))
# # # print(trainning_data_15dpd.count)
# #print data number
# # print(len(trainning_data_15dpd.index))

# #Turn Pandas Matrix into numpy array
# # print(trainning_data_15dpd.reset_index().values)
# # numpyMatrix = trainning_data_15dpd[:,1:10].as_matrix()
# # print(numpyMatrix)
# # print(type(numpyMatrix))

# # print(trainning_data_15dpd.iloc[0:5,1:10])

# # print(type(trainning_data_15dpd[0:5,1:10].values))

# #-----------
# #turn pandas: DataFrame ---> Numpy.array()
# #-----------
# # trainning_set = trainning_data_15dpd.as_matrix(columns= trainning_data_15dpd.columns[1:] ).astype(dtype = 'float32')
# # print(trainning_set)
# # print(type( trainning_set ))
# # print(trainning_set[1,1])
# # print(trainning_set.shape)

# #-----------------------------------------------------------------------------------------------------------------------------


'''------------------------------------------
[Algorithm] - linear regression
------------------------------------------'''


#usage: 
#data_set=numpy.array([])
def feature_normalize(data_set):
	print()

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





#usage:
# X = numpy.array([]) = np.array([x1,x2...xn])
# X_tmp = numpy.array([]) = np.array([x1,x2...xn, 1 ])
# y = numpy.array([]) = np.array([y1,y2...yn])
# theta = numpy.array([]) = np.array([w1,w2...wn, b ])
#learning_rate = float
#epochs = int

# def gradient_descent(X, y, X_validation, y_validation,theta, learning_rate, epochs):
# 	#-------------------------------------------------
# 	# Performs gradient descent to learn theta
# 	# by taking epochs gradient steps with learning_rate
# 	#-------------------------------------------------
# 	X_tmp = X
# 	n = X.shape[0]
# 	bias_ones = np.ones( (n,1) , dtype='float32')
# 	X_tmp = np.append(X, bias_ones, axis=1)

# 	X_validation_tmp = X_validation
# 	X_validation_tmp = np.append(X, bias_ones, axis=1)

# 	cost_history = np.zeros(shape=(epochs, 1))

# 	for i in range(epochs):
# 		predictions = X_tmp.dot(theta)
# 		theta_size = theta.shape[0]

# 		for it in range(theta_size):
# 			temp = X_tmp[:,it]
# 			temp.shape = (n,1)

# 			error_x1 = ( y - predictions) * (-temp)
# 			# error_x1 = ( y - predictions) * (temp)

# 			theta[it] = theta[it] - learning_rate * (1.0 / n) * (error_x1.sum())
# 		print("iteration numbers : " + str(i))
# 		# print("theta : " + str(theta))
# 		print("compute_cost : " + str( compute_cost(X, y, theta)))
# 		print(" [cost] - validation data : " + str( compute_cost(X_validation, y_validation, theta)))
# 		cost_history[i, 0] = compute_cost(X, y, theta)

# 	return theta, cost_history


#****************
#-----adagrad-----
#****************

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
	cost_history = np.zeros(shape=(epochs, 1))

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
			theta[it] = theta[it] - learning_rate * g/adagrad_coeff
		print("iteration numbers : " + str(i))
		# print("theta : " + str(theta))
		print(" [cost] - trainning data   : " + str( compute_cost(X, y, theta)))
		print(" [cost] - validation data : " + str( compute_cost(X_validation, y_validation, theta)))
		cost_history[i, 0] = compute_cost(X, y, theta)

	return theta, cost_history



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


	cost_history = np.zeros(shape=(epochs, 1))

	for i in range(epochs):
		predictions = X_tmp.dot(theta)
		theta_size = theta.shape[0]

		for it in range(theta_size):
			temp = X_tmp[:,it]
			temp.shape = (n,1)

			error_x1 = ( y - predictions) * (-temp)
			g = (1.0/n) * error_x1.sum()
			regularization_component = 2*regular_coeff*theta.sum()
			# error_x1 = ( y - predictions) * (temp)

			theta[it] = theta[it] - learning_rate * ( g + regularization_component )
		print("iteration numbers : " + str(i))
		# print("theta : " + str(theta))
		print(" [cost] - trainning data   : " + str( compute_cost_regularization(X, y, theta, regular_coeff)))
		print(" [cost] - validation data : " + str( compute_cost_regularization(X_validation, y_validation, theta, regular_coeff)))
		cost_history[i, 0] = compute_cost_regularization(X, y, theta, regular_coeff)

	return theta, cost_history

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

def trimming(X):
	print()

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
	print("trainning_set shape:" + str(trainning_set[:,0:10].shape))
	trainning_feature = trainning_set[:,0:9]
	trainning_yhat = trainning_set[:,9:10]
	validation_feature = validation_set[:,0:9]
	validation_yhat = validation_set[:,9:10]
	theta = np.random.rand(1,10).T - 0.5
	epochs = 1000
	learning_rate = 0.2

	# print(trainning_set)
	# print(trainning_feature)
	# print(trainning_yhat)
	# gradient_descent(trainning_feature, trainning_yhat, theta, 0.0001,1500)
	(theta,cost) = gradient_descent(trainning_feature, trainning_yhat,validation_feature,validation_yhat, theta, learning_rate ,epochs)
	# (theta_regular,cost_regular) = gradient_descent_regularization(trainning_feature, trainning_yhat,validation_feature,validation_yhat, theta, learning_rate,epochs, 0.01)
	predictions = testing(testing_set,theta)
	# print(predictions)
	# print(predictions.shape)
	write_file(predictions,"linear_regression.csv")
	# print("[normal] cost histroy :" + str(cost))
	# print("[regular] cost history :" + str(cost_regular))
