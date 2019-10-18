'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from math import sqrt
from numpy import log
import time
from datetime import timedelta

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



def sigmoid(z):
    zz = np.asarray(z)
    sigmoidValue = 1.0 / (1.0 + np.exp(-1.0 * zz))
    return  sigmoidValue

def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    
    y = []
    for label in training_label:
        y.append([1 if i == label else 0 for i in range(n_class)])
    y = np.array(y)
    #print(np.shape(y))
    n = training_data.shape[0] # 50000
    addBiasNodeInput = np.hstack((training_data, np.ones((n,1))))
    hiddenOutValue = sigmoid(addBiasNodeInput.dot(w1.T))
    addBiasNodeHidden = np.hstack((hiddenOutValue, np.ones((n, 1))))
    o = sigmoid(addBiasNodeHidden.dot(w2.T)) 
    error_withoutRegularization = -np.sum(y*log(o)+(1-y)*log((1-o)))/n
    obj_val = error_withoutRegularization + lambdaval/(2*n)*(np.sum(w1*w1) + np.sum(w2*w2))
    #1/(1+np.exp(-1*training_data.dot(w1)))
    #obj_val = 0.5*(y - output)**2
    deltal = o-y #########
    w1_gradient = np.dot(((1-hiddenOutValue)*(hiddenOutValue)*np.dot(deltal, np.delete(w2, -1, 1))).T, addBiasNodeInput)/n
    w2_gradient = np.dot(deltal.T, addBiasNodeHidden)/n
    w1_gradient_regularization = w1_gradient + (lambdaval/n)*w1
    #print(np.shape(w1_gradient_regularization))
    w2_gradient_regularization = w2_gradient + (lambdaval/n)*w2
    w1_gradient_regularization = w1_gradient_regularization.flatten()
    #print(np.shape(w1_gradient_regularization))
    w2_gradient_regularization = w2_gradient_regularization.flatten()

    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    obj_grad = np.concatenate((w1_gradient_regularization, w2_gradient_regularization),0)

    return obj_val, obj_grad

    
def nnPredict(w1,w2,data):
    labels = np.array([])
    n = data.shape[0]
    addBiasNodeInput = np.hstack((data, np.ones((n,1))))
    hiddenOutValue = sigmoid(addBiasNodeInput.dot(w1.T))
    addBiasNodeHidden = np.hstack((hiddenOutValue, np.ones((n, 1))))
    o = sigmoid(addBiasNodeHidden.dot(w2.T))
    labels = np.argmax(o, 1)
    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

# Start-time used for printing time-usage below.
start_time = time.time()

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

# Ending time.
end_time = time.time()

# Difference between start and end-times.
time_dif = end_time - start_time

# Print the time-usage.
print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

print('\n Number of nodes in hidden layer is: ' + str(n_hidden) + '  Lambda value is: ' + str(lambdaval))
#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')



# print("--- %s seconds ---" % (time.time() - start_time))