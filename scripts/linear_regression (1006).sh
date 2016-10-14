#!/usr/bin/env python
import numpy as np 
import pandas as pd 
import scipy 

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
# Extract Feature
#-------

print(trainning_data[:20])
#list component column
print(trainning_data[['location','component','0','5']])
# #list all PM2.5 component
# # print(trainning_data[trainning_data.component == "PM2.5"])
# #[high class]  list PM2.5 component 
is_PM25 = trainning_data['component'] =="PM2.5"
# # print(trainning_data[is_PM25] ['20140101':'20140220'])
print(trainning_data[is_PM25 ] [['component', '0','1','2' ]] ['20140101':'20140220'])

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


# print(trainning_data_15dpd)
# print(trainning_data_15dpd[0:5])
# print(trainning_data_15dpd.iloc[0:6,1:10])
#-----------
#turn pandas: DataFrame ---> Numpy.array()
#-----------
trainning_set_tmp = trainning_data_15dpd.as_matrix(columns= trainning_data_15dpd.columns[1:] ).astype(dtype = 'float32')
print(trainning_set_tmp)
print(type( trainning_set_tmp ))
print(trainning_set_tmp[1,1])


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
print("trainning set")
print(trainning_set)
print(trainning_set[1,:])
print(trainning_set[:,1])
print("trainning set shape         :" + str(trainning_set.shape))
print("Total number of data      :" + str(trainning_set_tmp.shape[0]))
print("number of rainning set   :" + str(trainning_set.shape[0]))
print("number of validation set:" + str(validation_set.shape[0]))




'''
For Testing
'''
#-----------------------------------------------------------------------------------------------------------------------------


# print(type(trainning_data_15dpd.iloc[0,5]))
# print(int(trainning_data_15dpd.iloc[0,5]))
# print(type(int(trainning_data_15dpd.iloc[0,5])))
# # print(trainning_data_15dpd.count)
#print data number
# print(len(trainning_data_15dpd.index))

#Turn Pandas Matrix into numpy array
# print(trainning_data_15dpd.reset_index().values)
# numpyMatrix = trainning_data_15dpd[:,1:10].as_matrix()
# print(numpyMatrix)
# print(type(numpyMatrix))

# print(trainning_data_15dpd.iloc[0:5,1:10])

# print(type(trainning_data_15dpd[0:5,1:10].values))

#-----------
#turn pandas: DataFrame ---> Numpy.array()
#-----------
# trainning_set = trainning_data_15dpd.as_matrix(columns= trainning_data_15dpd.columns[1:] ).astype(dtype = 'float32')
# print(trainning_set)
# print(type( trainning_set ))
# print(trainning_set[1,1])
# print(trainning_set.shape)

#-----------------------------------------------------------------------------------------------------------------------------


'''
[Algorithm] - linear regression
'''
learning_rate = 0.2
#lambda for regularization 
regularization_parameter = 0.1

#usage: 
#data_set=numpy.array([])
def feature_normalize(data_set):
	print()

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

#usage:
# X = numpy.array([]) = np.array([x1,x2...xn])
# X_tmp = numpy.array([]) = np.array([x1,x2...xn, 1 ])
# y = numpy.array([]) = np.array([y1,y2...yn])
# theta = numpy.array([]) = np.array([w1,w2...wn, b ])
#learning_rate = float
#num_iters = int

# def gradient_descent(X, y, theta, learning_rate, num_iters):
# 	#-------------------------------------------------
# 	# Performs gradient descent to learn theta
# 	# by taking num_iters gradient steps with learning_rate
# 	#-------------------------------------------------
# 	X_tmp = X
# 	n = X.shape[0]
# 	bias_ones = np.ones( (n,1) , dtype='float32')
# 	X_tmp = np.append(X, bias_ones, axis=1)

# 	cost_history = np.zeros(shape=(num_iters, 1))

# 	for i in range(num_iters):
# 		predictions = X_tmp.dot(theta)
# 		theta_size = theta.shape[0]

# 		for it in range(theta_size):
# 			temp = X_tmp[:,it]
# 			temp.shape = (n,1)

# 			partial_theta = ( y - predictions) * (-temp)
# 			# partial_theta = ( y - predictions) * (temp)

# 			theta[it] = theta[it] - learning_rate * (1.0 / n) * (partial_theta.sum())
# 		print("iteration numbers : " + str(i))
# 		# print("theta : " + str(theta))
# 		print("compute_cost : " + str( compute_cost(X, y, theta)))
# 		cost_history[i, 0] = compute_cost(X, y, theta)

# 	return theta, cost_history

def gradient_descent(X, y, X_validation, y_validation, theta, learning_rate, num_iters):
	#-------------------------------------------------
	# Performs gradient descent to learn theta
	# by taking num_iters gradient steps with learning_rate
	#-------------------------------------------------
	X_tmp = X
	n = X.shape[0]
	bias_ones = np.ones( (n,1) , dtype='float32')
	X_tmp = np.append(X, bias_ones, axis=1)

	X_validation_tmp = X_validation
	X_validation_tmp = np.append(X, bias_ones, axis=1)


	cost_history = np.zeros(shape=(num_iters, 1))

	for i in range(num_iters):
		predictions = X_tmp.dot(theta)
		theta_size = theta.shape[0]

		for it in range(theta_size):
			temp = X_tmp[:,it]
			temp.shape = (n,1)

			partial_theta = ( y - predictions) * (-temp)
			# partial_theta = ( y - predictions) * (temp)

			theta[it] = theta[it] - learning_rate * (1.0 / n) * (partial_theta.sum())
		print("iteration numbers : " + str(i))
		# print("theta : " + str(theta))
		print(" [cost] - trainning data   : " + str( compute_cost(X, y, theta)))
		print(" [cost] - validation data : " + str( compute_cost(X_validation, y_validation, theta)))
		cost_history[i, 0] = compute_cost(X, y, theta)

	return theta, cost_history




if __name__ == '__main__':
	print("trainning_set shape:" + str(trainning_set[:,0:10].shape))
	trainning_feature = trainning_set[:,0:9]
	trainning_yhat = trainning_set[:,9:10]
	validation_feature = validation_set[:,0:9]
	validation_yhat = validation_set[:,9:10]
	theta = np.random.rand(1,10).T - 0.5
	# print(trainning_set)
	# print(trainning_feature)
	# print(trainning_yhat)
	# gradient_descent(trainning_feature, trainning_yhat, theta, 0.0001,1500)
	gradient_descent(trainning_feature, trainning_yhat,validation_feature,validation_yhat, theta, 0.0001,1500)

