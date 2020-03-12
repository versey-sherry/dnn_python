from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import time

np.random.seed(914)

#default learning rate 0.1, epoch 100
def dnn(h1_node=9, h2_node=10, epoch =100, lr = 0.1):
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
    #normalize the dataset
    def norm(x):
        return (x - np.min(x))/(np.max(x)-np.min(x))

    #Loading the dataset
    iris = datasets.load_iris()
    #ndarrays, data has 4 attributes, target has 3 labels
    original_data = iris.data
    original_data= np.apply_along_axis(norm, 0, original_data)
    #Adding bias term
    data = np.insert(original_data, 4, 1, axis= 1)
    original_target = iris.target

    #Train_test split
    X_train, X_test, y_train, y_test = train_test_split(data, original_target, test_size = 0.2, random_state = 914)
    y_train = np.transpose(np.array([np.where(y_train ==0,1,0), np.where(y_train ==1,1,0), np.where(y_train ==2,1,0)]))

    #Initialize the weights (nodes, weights)
    #input-h1
    w01 = np.random.rand(h1_node, 5)/2
    #h1-h2
    w12 = np.random.rand(h2_node,h1_node+1)/2
    #h2-ouput
    w2o = np.random.rand(3, h2_node+1)/2

    #iterating through the epochs
    for i in range(100):
        for item in range(X_train.shape[0]):
            data = X_train[item, :]
            layer01 = np.dot(w01, np.transpose(data))
            h1 = sigmoid(layer01)
            h1 = np.hstack((h1, np.ones(1)))
            #print(h1)
            layer12 = np.dot(w12, h1)
            h2 = sigmoid(layer12)
            h2 = np.hstack((h2, np.ones(1)))
            #print(h2)
            layer2o = np.dot(w2o, h2)
            output = sigmoid(layer2o)
            loss = np.sum(np.square(output - y_train[item,:]))
            #backprop
            del3 = (output - y_train[item, :]) * (output*(1-output))
            delta_w2o = np.dot(np.reshape(del3, (3,1)), np.reshape(np.transpose(h2), (1,h2_node+1)))
            del2 = np.dot(np.transpose(w2o), del3)[0:h2_node, ] * (h2*(1-h2))[0:h2_node, ]
            delta_w12 = np.dot(np.reshape(del2, (h2_node, 1)), np.reshape(np.transpose(h1),(1, h1_node +1)))
            del1 = np.dot(np.transpose(w12), del2)[0:h1_node,] *(h1*(1-h1))[0:h1_node,]
            delta_w01 = np.dot(np.reshape(del1, (h1_node,1)),np.reshape(np.transpose(data),(1, 5)))
            #Weight updating
            w2o = w2o - lr*delta_w2o
            w12 = w12 -lr*delta_w12
            w01 = w01 - lr*delta_w01

        #predict
        layer01 = np.dot(w01, np.transpose(X_test))
        h1 = sigmoid(layer01)#9*5 dot 5*150 = 9 *150
        h1 = np.vstack((h1, np.ones(h1.shape[1])))
        layer12 = np.dot(w12, h1)#10*10 dot 10*150 =10*150
        h2 = sigmoid(layer12)
        h2 = np.vstack((h2, np.ones(h2.shape[1])))
        layer2o = np.dot(w2o, h2) #3*11 dot 11*150 = 3*150
        output = sigmoid(layer2o)
        sum(np.argmax(output, axis = 0) == y_test)/len(y_test)

    print("accuracy is", sum(np.argmax(output, axis = 0) == y_test)/len(y_test))



def main():
    node_number = input("Input node numbers for the two hidden layers, eg. 10 10: ")
    epoch = int(input("Input your desired epoch number, eg. 100: "))
    lr = float(input("Input your desired learning rate, eg. 0.2: "))
    h1_node = int(node_number.split()[0])
    h2_node = int(node_number.split()[1])
    now = time.time()
    dnn(h1_node, h2_node, epoch, lr)
    print("Time difference is", time.time()-now, "seconds")

if __name__ == '__main__':
    main()

