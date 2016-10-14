import matplotlib.pyplot as plt 

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
