# code to write nodes for neual network
import numpy as np
import sys
import matplotlib as plt
import math

int_layer = [1,2,3]
weight_matrix = np.array([[1,2,3,4],
                          [5,6,7,8],
                          [9,10,11,12]])


# matrix multiplication 
def sigmoid(X):
    return np.exp(X)/(1+np.exp(X))

class layer:
    def __init__(self,lt,weight_matrix,biases):
        self.len = len(lt)
        self.w = weight_matrix
        self.l = lt
        self.b = biases
    def forward(self):
        weight_matrix = self.w
        output_layer = []
        for j in range (np.shape(weight_matrix)[1]):
            output = 0
            for i in range (self.len):
                output += weight_matrix[i,j]*self.l[i] 
            output += self.b[j]
            output_layer.append(output)
        return sigmoid(output_layer)
    # def backward(self)


int_layer = layer(int_layer,weight_matrix,[-1,0,1,2])
print(int_layer.forward())
    
# class node:

    