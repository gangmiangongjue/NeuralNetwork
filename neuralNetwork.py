import numpy as np
from sigmoid import Sigmoid
class NeuralNetwork:
    def __init__(self,x,y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output == np.zeros(y.shape)
    def feedforward(self):
        self.layer1 = Sigmoid.dot(self.input,self.weights1)
        self.output = Sigmoid.dot(self.layer1,self.weights2)
    def backprop(self):
        d_weight2 = np.dot(self.layer1.T,(2*(self.y-self.output)*Sigmoid.derivative(self.output)))
        d_weight1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * Sigmoid.derivative(self.output), self.weights2.T) * Sigmoid.derivative(self.layer1)))
        self.weights2 = d_weight2
        self.weights1 = d_weight1