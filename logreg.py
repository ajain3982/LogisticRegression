import pandas as pd
import numpy as np
import argparse

class LogisticRegression:
    '''
    Class to accomodate the code
    '''


    def __init__(self,learning_rate=0.01,epochs=10,initialiser=None,verbose=1):
        '''
        this method initialised the required parameters
        '''
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.initialiser = initialiser
        self.verbose = verbose

    def initialise(self,num_weights,initialiser=None):

        '''
        Based on the initialiser value, this method initialises the weights.
        '''
        if initialiser == None or initialiser=='random':
            nn_weights = np.random.randn(num_weights)

        elif initialiser == 'he_normal':
            nn_weights = np.random.randn(1,num_weights)*np.sqrt(2/(num_weights+1))

        elif initialiser == "xavier":
            nn_weights = np.random.randn(1,num_weights)*np.sqrt(1/num_weights)

        return nn_weights

    def sigmoid(self,X):
        '''
        A function to calculate sigmoid
        '''
        return 1 / (1 + np.exp(-X))

    def forward_prop(self,input,weights):

        '''
        method to do forward propagation
        '''
        z = np.dot(input,weights)
        output = self.sigmoid(z)

        return output

    def binary_cross_entropy(self,y_true,y_pred):
        '''
        This function calculates binary cross entropy loss
        '''
        loss = (- y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)).mean()

        return loss

    def calculate_gradient(self,input,y_true,y_pred):

        '''
        This function calculates the gradients
        '''
        error = y_pred - y_true
        gradient = np.dot(input.T,error) / y_true.shape[0]

        return gradient

    def back_prop(self,input,y_true,y_pred,old_weights):
        '''
        method to do back propagation and learn weights via gradient descent
        '''
        gradient = self.calculate_gradient(input,y_true,y_pred)
        new_weights = old_weights - self.learning_rate * gradient

        return new_weights

    def fit(self,X,y):

        '''
        This method loops for number of epochs and learns the weights
        '''

        self.weights = self.initialise(X.shape[1],self.initialiser)
        for i in range(0,self.epochs):
            output = self.forward_prop(X,self.weights)
            loss = self.binary_cross_entropy(y,output)
            self.weights = self.back_prop(X,y,output,self.weights)

            if self.verbose:
                print("Epoch Number : " + str(i) + ", Loss: " + str(loss))

    def predict(self,X):
        '''
        This method takes in 2d array and gives out the predictions
        '''
        output = self.forward_prop(X,self.weights)
        return (output >= 0.5)
