from sklearn import datasets
import numpy as np

#x is z from previous layers
def sigmoid(x):
	return 1/(1+np.exp(-x))
#array of predict and array of target
def mean_sqared_error(predict, target):
	return np.sum(np.square(a-b))
#array of exp(z)
def softmax(x):
	return max(x)/np.sum(x)

#Loading the dataset
iris = datasets.load_iris()
#ndarrays, data has 4 attributes, target has 3 labels
data = iris.data
target = iris.target

#There are 6 layers. Input layer 4 nodes, h1 4 nodes, h2 4 nodes, h3 4 nodes, h4 4 nodes output 3 nodes. Matching dimensions for cannon's algorithm
#TensorFlow output Test loss: 0.11755118042230606, Test accuracy: 0.92, Time difference: 4.986849069595337
#This is be implemented with full gradient descent

#Initialize the weights
#input-h1
w01 = np.random.rand(4,4)/2
#h1-h2
w12 = np.random.rand(4,4)/2
#h2-h3
w23 = np.random.rand(4,4)/2
#h3-h4
w34 = np.random.rand(4,4)/2
#h4-h5
w4o = np.random.rand(4,3)/2

#Learning rate
lr = 0.1


