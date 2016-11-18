#!/usr/bin/env python
import numpy as np 
import pickle
import scipy
import math
import time
import theano
import keras
import sys
import matplotlib.pyplot as plt 

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.models import load_model

# from keras.layers.advanced_activation import LeakyReLU, PReLU

'''
Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train.py
'''

def read_data(path):
	tStart = time.time()

	# input image dimensions
	img_rows, img_cols = 32, 32
	# the CIFAR10 images are RGB
	img_channels = 3

	print("Reading labeled data")
	#label data : X_train_label , y_train_label
	all_label_filename =  path +  "all_label.p"
	all_label = pickle.load(open(all_label_filename , 'rb'))
	all_label = np.array(all_label)

	#input image total class
	num_class = len(all_label)
	#input image total numver
	num_image = len(all_label[0])
	#total image nb
	nb_total_image = num_class * num_image

	#labeled image data structure declare
	X_train_label = np.zeros(shape=(nb_total_image,img_channels,img_rows,img_cols )).astype(dtype = 'float32')
	y_train_label = np.array([0] * num_class * num_image).astype(dtype='int').reshape(num_class * num_image,1)

	#feature
	for i in range(num_class):
		for j in range(num_image):
			tmp = np.array( all_label[i][j] ).astype(dtype='float32').reshape(1,img_channels,img_rows,img_cols)
			X_train_label[ i*num_image + j ] = tmp

		print("reading labeled data :" + str(500*i))



	#label
	for i in range(num_class):
		for j in range(num_image):
			y_train_label[ i*num_image + j] = i

	# np.save("X_train_label", X_train_label)
	# np.save("y_train_label",y_train_label)




	#unlabel data : X_train_unlabel
	all_unlabel_filename =  path + "all_unlabel.p"
	all_unlabel = pickle.load(open(all_unlabel_filename , 'rb'))
	all_unlabel = np.array(all_unlabel)


	#input image total number
	num_image = len(all_unlabel)
	# input image dimensions
	img_rows, img_cols = 32, 32
	# the CIFAR10 images are RGB
	img_channels = 3


	#unlabel data structure declare :X_train_unlabel
	X_train_unlabel = np.zeros(shape = (num_image,img_channels,img_rows,img_cols) ).astype(dtype='float32')

	for i in range(num_image):
		tmp = np.array(all_unlabel[i]).astype(dtype='float32').reshape(img_channels,img_rows,img_cols)
		X_train_unlabel[i] = tmp

		if(i % 1000 == 0):
			print("reading ulabel data :" + str(i))

	# print("X_train_unlabel.shape" + str( X_train_unlabel.shape ))

	# np.save("X_train_unlabel", X_train_unlabel)



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


	tEnd = time.time()
	cost_time = tEnd - tStart
	print("Reading data cost " + str(cost_time) + "sec")

	return X_train_label,y_train_label,X_train_unlabel,X_test,ID_test 



def add_unlabel_featrue2training(training_feature, training_yhat, unlabel_feature, model, confidence_th):
	unlabel_prediction = model.predict(unlabel_feature)
	# print("nb_unlabel_feature:" + str(len(unlabel_feature) ))
	# print("nb_unlabel_prediction:" + str(len(unlabel_prediction)))
	nb_unlabel_data = len(unlabel_prediction)

	delete_array = np.array([0]*nb_unlabel_data).astype(dtype='int')
	delete_index = np.array([]).astype(dtype='int')
	count = 0

	print("[ original data ] : training->" + str(len(training_feature)) + "  unlabel->"+str(len(unlabel_feature)) )



	for i in range(nb_unlabel_data):
		if(  np.amax( unlabel_prediction[i] ) >= confidence_th  ):

			delete_array[i] = 1
			max_index = np.argmax(unlabel_prediction[i])
			tmp_prediction = unlabel_prediction[i]
			tmp_prediction.fill(0)
			tmp_prediction[max_index] = 1.0

			tmp_feature =  unlabel_feature[i]
			tmp_feature = np.array( tmp_feature ).astype(dtype='float32').reshape(1,3,32,32)
			training_feature = np.vstack((training_feature, tmp_feature ))


			tmp_prediction = np.array( tmp_prediction ).astype(dtype='float32').reshape(1,10)			
			training_yhat     = np.vstack((training_yhat, tmp_prediction ))

			count += 1
		if(i%1000 == 0 ):
			print("check unlabel confidence :" + str(i) +"  data")


	for i in range( nb_unlabel_data):
		if(delete_array[i] == 1):
			delete_index = np.append(delete_index, i )	
		if(i%1000 == 0 ):
			print("delete unlabel feature :" + str(i) +"  data")

	unlabel_feature = np.delete(unlabel_feature, delete_index, axis = 0 )	


	print("after confidence similar trimming:")
	print("total move :" + str(count) + "  data from unlabel to training ")
	print("[after data] : training->" + str(len(training_feature)) + "  unlabel->" + str(len(unlabel_feature)) )



	return training_feature, training_yhat, unlabel_feature





def split_data(training_feature, training_yhat, validation_percent):
	
	training_percent = 1 - validation_percent 
	num_training = int(training_percent * training_feature.shape[0])
	indices = np.random.permutation(training_feature.shape[0])
	training_idx,validation_idx = indices[:num_training], indices[num_training:]
	# print("training_feature.shape[0]",training_feature.shape[0])
	# print()

	training_feature ,validation_feature = training_feature[training_idx,:], training_feature[validation_idx,:]
	training_yhat ,validation_yhat = training_yhat[training_idx,:], training_yhat[validation_idx,:]

	return training_feature , training_yhat , validation_feature , validation_yhat



def create_model():

	'''
	Keras model
	'''
	batch_size = 100
	nb_classes = 10
	nb_epoch = 500

	nb_conv_filter = 48

	# input image dimensions
	img_rows, img_cols = 32, 32
	# the CIFAR10 images are RGB
	img_channels = 3

	#Step 1 : model->define a set of function
	model = Sequential()

	#border_mode ='same' : output size = input size ,due to the zero padding .
	#border_mode = 'valid' : output size is smaller input size. since only calculate fully occupy.
	model.add(Convolution2D(2*nb_conv_filter , 3,3 , border_mode='same',
			input_shape = (img_channels, img_rows, img_cols) ) )
	model.add(Activation('relu'))
	model.add(Dropout(0.3))	

	model.add(Convolution2D(2*nb_conv_filter , 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Dropout(0.3))
	
	model.add(Convolution2D(2*nb_conv_filter , 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Dropout(0.3))

	
	# model.add(MaxPooling2D(pool_size=(3,3),strides=(1,2) ) )
	model.add(MaxPooling2D(pool_size=(3,3) ) )
	model.add(Dropout(0.3))

	model.add(Convolution2D(4*nb_conv_filter , 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Dropout(0.4))

	model.add(Convolution2D(4*nb_conv_filter , 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Dropout(0.4))

	model.add(Convolution2D(4*nb_conv_filter , 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Dropout(0.4))
	
	# model.add(Convolution2D(192, 1, 1, border_mode='same',))
	# model.add(Activation('relu'))
	# # model.add(Dropout(0.25))


	# model.add(MaxPooling2D(pool_size=(3,3),strides=(1,2) ))
	model.add(MaxPooling2D(pool_size=(3,3)))
	model.add(Dropout(0.4))

	model.add(Convolution2D(192, 3, 3, border_mode='same',))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Convolution2D(192, 1, 1, border_mode='same',))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Convolution2D(10, 1, 1, border_mode='same',))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Flatten())


	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	#Step 2 : goodness of function [loss function]
	#Step 3 : pick the best function [optimizer]

	#train the model using stochastic gradient decent(SGD) + momentum
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',
			optimizer='adam',
			metrics=['accuracy'])


	return model



if __name__ == "__main__":

	flag_read_data = True

	flag_supervised = False
	flag_semi_supervised = True


	'''
	reading data
	'''
	path = sys.argv[1]
	output_model = sys.argv[2]

	if flag_read_data == True:
		(X_train_label, y_train_label, X_train_unlabel, X_test,ID_test) = read_data(path)

		(training_feature , training_yhat , validation_feature , validation_yhat) = split_data(X_train_label , y_train_label , 0.1 )

		nb_classes = 10
		# convert class vectors to binary class matrices
		training_yhat = np_utils.to_categorical(training_yhat, nb_classes)
		validation_yhat = np_utils.to_categorical(validation_yhat, nb_classes)

		#normalized
		training_feature -= 128
		validation_feature -= 128
		training_feature /= 255 
		validation_feature /= 255


		# #unlabel data
		unlabel_feature = X_train_unlabel

		unlabel_feature -= 128
		unlabel_feature /= 255 

		#testing data
		test_feature = X_test

		test_feature -= 128
		test_feature /= 255





	'''
	Training
	'''

	batch_size = 100
	nb_classes = 10
	nb_epoch = 500

	nb_conv_filter = 48

	# input image dimensions
	img_rows, img_cols = 32, 32
	# the CIFAR10 images are RGB
	img_channels = 3
	#early stop
	early_stop = EarlyStopping(monitor='val_loss' , patience=20, verbose=1)



	#no unlabel
	if flag_supervised == True:
		model = create_model()
		model.fit(training_feature, training_yhat, batch_size = batch_size, nb_epoch = nb_epoch, shuffle=True, validation_data=(validation_feature, validation_yhat), callbacks=[early_stop] )


	#use unlabel
	if flag_semi_supervised == True:
		nb_add_unlabel = 5
		# nb_add_unlabel = 2
		confidence_th = 0.995

		for i in range(nb_add_unlabel):	

			if i == 	nb_add_unlabel -1:
				model = create_model()
				nb_epoch = 150
				# nb_epoch = 1
				hist = model.fit(training_feature, training_yhat, batch_size = batch_size, nb_epoch = nb_epoch, validation_data=(validation_feature, validation_yhat), callbacks=[early_stop] )
			else:
				model = create_model()
				nb_epoch = 100
				# nb_epoch = 1
				model.fit(training_feature, training_yhat, batch_size = batch_size, nb_epoch = nb_epoch, validation_data=(validation_feature, validation_yhat), callbacks=[early_stop] )
				(training_feature, training_yhat, unlabel_feature) = add_unlabel_featrue2training(training_feature, training_yhat, unlabel_feature, model, confidence_th)
				del(model)



	'''
	Save model
	'''
	model.save(output_model + '.h5')



	'''
	plot picture
	'''


	# plt.plot(hist.history['acc'])
	# plt.plot(hist.history['val_acc'])
	# plt.title('model accuracy')
	# plt.ylabel('accuracy')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'test'], loc='upper left')
	# plt.show()
	# # summarize history for loss
	# plt.plot(hist.history['loss'])
	# plt.plot(hist.history['val_loss'])
	# plt.title('model loss')
	# plt.ylabel('loss')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'test'], loc='upper left')
	# plt.show()
	




	# '''
	# calculate predictionss
	# '''
	# predictions_filename = "predictions.csv"
	# predictions_tmp = model.predict(test_feature)
	# nb_predictions = len(predictions_tmp)

	# predictions = np.array([0] * nb_predictions).astype(dtype='int')

	# for i in range( nb_predictions ):
	# 	predictions[i] = predictions_tmp[i].argmax()

	# print("predictions_tmp.shape : ",predictions_tmp.shape)
	# # predictionss = [round(x) for x in predictionss_tmp]

	# write_file(test_ID,predictions, predictions_filename)




