#!/usr/bin/env python
import numpy as np 
import pickle
import scipy
import math
import time
import theano
import keras
import sys

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.models import load_model


def read_data(path):

	img_channels = 3
	img_rows = 32
	img_cols = 32
	#testing data : X_test , ID_test
	test_filename = path+"test.p"
	test = pickle.load(open(test_filename,'rb') )
	data_test = test['data'][:]
	ID_test = test['ID'][:]

	data_test = np.array(data_test)
	ID_test = np.array(ID_test)

	np_image = len(data_test)

	X_test = np.zeros(shape = (np_image, img_channels,img_rows,img_cols) ).astype(dtype='float32')

	for i in range(np_image):
		tmp = np.array(data_test[i]).astype(dtype='float32').reshape(img_channels,img_rows,img_cols)
		X_test[i] = tmp

		if(i % 1000 == 0):
			print("reading testing data :" + str(i) )

	return X_test,ID_test

	# np.save("X_test", X_test)
	# np.save("ID_test",ID_test)



def write_file(ID,predictions,file_name):
	predictions.shape = (len(predictions) , 1)
	writer = open(file_name,"w")
	writer.write("ID,class\n")
	for i in range(len(predictions)):
		str_tmp = str( ID[i] )+"," + str(predictions[i][0]) + "\n"
		# print(str_tmp)
		writer.write(str_tmp)		
	writer.close()


if __name__ == "__main__":

	flag_read_data = True
	flag_load_data = True


	tStart = time.time()

	'''
	calculate predictionss
	'''
	path = sys.argv[1]
	model_name = sys.argv[2]
	predictions_filename = sys.argv[3]

	if flag_read_data == True:
		(test_feature,test_ID) = read_data(path)
		print("test_feature.shape" + str(test_feature.shape) )
		test_feature -= 128
		test_feature /= 255		

	# #testing data
	# if flag_load_data == True:
	# 	print("loading testing data:")
	# 	test_feature = np.load('X_test.npy')
	# 	test_ID = np.load('ID_test.npy')

	# 	test_feature -= 128
	# 	test_feature /= 255

	print("load model")
	model = load_model(model_name+'.h5')
	predictions_tmp = model.predict(test_feature)
	nb_predictions = len(predictions_tmp)

	predictions = np.array([0] * nb_predictions).astype(dtype='int')

	print("predict")
	for i in range( nb_predictions ):
		predictions[i] = predictions_tmp[i].argmax()

	# print("predictions_tmp.shape : ",predictions_tmp.shape)
	# predictionss = [round(x) for x in predictionss_tmp]

	print("write prediction.csv")
	write_file(test_ID,predictions, predictions_filename)


	tEnd = time.time()
	cost_time = tEnd - tStart
	print("Testing cost " + str(cost_time) + "sec")