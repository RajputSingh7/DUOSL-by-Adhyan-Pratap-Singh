# code to write nodes for neual network
import numpy as np
import sys
import matplotlib as plt
import math
import random as ran
import nnfs
from nnfs.datasets import spiral_data


# matrix multiplication 
class sigmoid:
    def forward(self,X):
        self.output = np.exp(X)/(1+np.exp(X))
        return self.output
class ReLu:
    def forward(self,X):
        self.output = np.maximum(0,X)
        return self.output
class softmax:
    def forward(self,X):
        exp = np.exp(X - np.max(X,axis = 1, keepdims = True))
        probs = exp / np.sum(exp,axis = 1,keepdims = True)
        self.output = probs
        return self.output
# common loss class
class loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses) # sample losses is array 1D for loss at each node of output
class CategoricalCrossentropy(loss):
    def forward(self,y_pred,y_actual):
        samples = len(y_pred)
        y_pred_clip = np.clip(y_pred,1e-7,1 - 1e-7) # for eliminating the zero division error
        if len(y_actual.shape)== 1:
            correct_confidences = y_pred_clip[range(samples),y_actual]
        elif len(y_actual.shape) == 2:
            correct_confidences = np.sum(y_pred_clip*y_actual , axis = 1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
X,y = spiral_data(samples = 10,classes = 4)
print(X)
class layer:
    def __init__(self,n,m):
        self.m = m
        self.n = n
        self.w = np.random.rand(self.n,self.m)
        self.b = np.zeros(m)
        print(self.w)
        print(self.b)
    def forward(self,lt):
        weight_matrix = self.w
        self.l = lt
        output_layer = []
        for j in range (self.m):
            output = 0
            for i in range (self.n):
                output += weight_matrix[i,j]*self.l[i] 
            output += self.b[j]
            output_layer.append(output)
        output_layer = np.array(output_layer)
        Rel = softmax()
        self.output = Rel.forward(output_layer)
    # def backward(self)


int_layer = layer(4,3)
layer2 = layer(3,5)
int_layer.forward(X)
layer2.forward(int_layer.output)
print(layer2.output)
    

    