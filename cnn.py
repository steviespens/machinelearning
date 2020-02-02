# baseline cnn model for mnist
from keras.applications.vgg16 import VGG16
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

import numpy as np
# load train and test dataset


def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels


def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model




def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(Conv2D(16, (6, 6), activation='relu', kernel_initializer='he_uniform', input_shape=(26,26,32)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# evaluate a model using k-fold cross-validation


def evaluate_model(model, dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=1, batch_size=32,
		                    validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories

# plot diagnostic learning curves


def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(211)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(212)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'],
		            color='orange', label='test')
	pyplot.show()

# summarize model performance


def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' %
	      (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	# pyplot.show()

# run the test harness for evaluating a model


def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model()
	# evaluate model
	scores, histories = evaluate_model(model, trainX, trainY)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)


# entry point, run the test harness
# run_test_harness()


def driver():
	# load dataset

    trainX, trainY, testX, testY = load_dataset()
    trainN = 10000
    testN = 1000
    trainX = trainX[:trainN]
    trainY = trainY[:trainN]
    testX = testX[:testN]
    testY = testY[:testN]

	# prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)

	# # define model
    model = define_model()
	# # evaluate model

    history = model.fit(trainX, trainY, epochs=10, batch_size=32,
                        validation_data=(testX, testY), verbose=0)

	# scores, histories = evaluate_model(model, trainX, trainY)
	# # learning c
    # urves
	# summarize_diagnostics(histories)
	# # summarize estimated performance
	# summarize_performance(scores)

def plot(img):
    for i in range(len(img)):

        pyplot.subplot(len(img)//5, 5, i+1)
        pyplot.imshow(np.squeeze(img[i]), cmap=pyplot.get_cmap('gray'))
    pyplot.show()


def pr(a):
    
    for d,c in enumerate(a):
        print(d)
        for i, j in enumerate(c):
            print(i, j)
        print('\n')

def predict(model, x):
    return model.predict(x)

trainX, trainY, testX, testY = load_dataset()
trainN = 100
testN = 100
trainX = trainX[50:50+trainN]
trainY = trainY[50:50+trainN]
testX = testX[:testN]
testY = testY[:testN]

# prepare pixel data
trainX, testX = prep_pixels(trainX, testX)

# # define model
model = define_model()
# # evaluate model

history = model.fit(trainX, trainY, epochs=1, batch_size=32,
                    validation_data=(testX, testY), verbose=0)

n=15
x = testX[:n]
y = testY[:n]

a = model.predict(x)
# _, acc = model.evaluate(x, y, verbose=0)
a, b = evaluate_model(model, testX, testY)
summarize_performance(a)

#VISUALIZE FILTERS
# retrieve weights from the second hidden layer
filters, biases = model.layers[0].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(1):
		# specify subplot and turn of axis
		ax = pyplot.subplot(n_filters, 1, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
pyplot.show()


# for l in model.layers:
#     print(l.output_shape)
print('done')


# 50.000
# > 65.000
# > 80.000
# > 85.000
# > 100.000

# > 55.000
# > 50.000
# > 65.000
# > 90.000
# > 95.000


# > 40.000
# > 60.000
# > 80.000
# > 90.000
# > 100.000
