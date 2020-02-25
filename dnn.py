from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler

np.random.seed(914)

#x is z from previous layers
def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.where(x<0,0,x)
#array of predict and array of target
def mean_sqared_error(predict, target):
    return np.sum(np.square(predict - target), axis = 0)
#array of exp(z)
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis =0)

#Loading the dataset
iris = datasets.load_iris()
#ndarrays, data has 4 attributes, target has 3 labels
original_data = iris.data
#Adding bias term
data = StandardScaler().fit_transform(original_data)
data = np.insert(data, 4, 1, axis= 1)
original_target = iris.target
target = np.array([np.where(original_target ==0,1,0), np.where(original_target ==1,1,0), np.where(original_target ==2,1,0)])

#There are 5 layers. Input layer 10 nodes, h1 10 nodes, h2 10 nodes, h3 10 nodes, output 3 nodes. Matching dimensions for cannon's algorithm
#Test loss: 0.04626183790465196, Test accuracy: 0.9333333333333333, Time difference: 5.915765047073364

#Initialize the weights (nodes, weights)
#input-h1
w01 = np.random.rand(9, 5)/2
#h1-h2
w12 = np.random.rand(10,10)/2
#h2-ouput
w3o = np.random.rand(3, 11)/2

#Learning rate
lr = 0.1

#Forward propogation
layer01 = np.dot(w01, np.transpose(data))
h1 = sigmoid(layer01)#9*5 dot 5*150 = 9 *150
h1 = np.vstack((h1, np.ones(h1.shape[1])))
layer12 = np.dot(w12, h1)#10*10 dot 10*150 =10*150
h2 = sigmoid(layer12)
h2 = np.vstack((h2, np.ones(h2.shape[1])))
layer2o = np.dot(w2o, h2) #3*11 dot 11*150 = 3*150
output = sigmoid(layer2o)
loss = np.sum(np.square(output - target))
print(loss)

#backward (minus delta)
batch_size = 150
delta_w2o = np.dot(((output - target) * ((-np.exp(-layer2o)*output)/np.square(1-np.exp(-layer2o)))), np.transpose(h2))/batch_size
#delta_w12
delta_w12 =np.dot(np.transpose(w2o), ((output - target) * ((-np.exp(-layer2o)*output)/np.square(1-np.exp(-layer2o)))))
#delete the one row that doesn't come from calculation
delta_w12 = delta_w12[0:10,:]
intermediate = delta_w12 * ((-np.exp(-layer12)*sigmoid(layer12))/np.square(1-np.exp(-layer12)))
delta_w12 = np.dot(intermediate, np.transpose(h1))/batch_size
#delta_w01
delta_w01 =np.dot(np.transpose(w12), intermediate)
delta_w01 = delta_w01[0:9,:]
delta_w01 = np.dot(delta_w01 * ((-np.exp(-layer01)*sigmoid(layer01))/np.square(1-np.exp(-layer01))), data)/batch_size
#Weight updating
w2o = w2o + lr*delta_w2o
w12 = w12 +lr*delta_w12
w01 = w01 + lr*delta_w01
