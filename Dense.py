import numpy as np
import math
from BP_fromScratch import Adam

class Dense:

    def __init__(self, units, activation_func):
        self.units = units
        self.size = units
        self.activation_type = activation_func 

    def build(self, input_size, set_adam, eta):
        #random_values = np.random.normal(0, 1, size=input_size*self.units) # Gaussian initialization
        # Xavier initialization
        random_values = np.random.uniform(-np.sqrt(6/(input_size+self.units)), np.sqrt(6/(input_size+self.units)), size=input_size * self.units)
        self.w = random_values.reshape(input_size, self.units)
        self.b = np.random.normal(0, 1, size=self.units).reshape(1, self.units)
        self.input_size = input_size

        if set_adam:
            self.adam = Adam.Adam(input_size, self.units, eta, beta1=0.9, beta2=0.999, epsilon=1e-8)

    def call_act(self, inputs):
        return np.array(self.activation(np.matmul(inputs, self.w) + self.b)) # y=f(s)=h
    
    def call(self, inputs):
        return np.array(np.matmul(inputs, self.w) + self.b) # s
        
    def activation(self, input):

        if self.activation_type == "relu":
            return np.maximum(np.zeros(input.size), input)
        elif self.activation_type == "linear":
            return input
        elif self.activation_type == "sigmoid":
            return self.sigmoid(input)
        elif self.activation_type == "softmax":
            return self.softmax(input)

    def activation_deriv(self, input):

        if self.activation_type == "linear":
            return np.ones(input.size)
        elif self.activation_type == "relu":
            a = np.zeros(input.size)
            for i in range(input.size):
                if input[0,i]>=0:
                    a[i]=1
            return a
        elif self.activation_type =="sigmoid":
            return self.sigmoid_deriv(input)
        elif self.activation_type =="softmax":
            return self.softmax_deriv(input)

    def sigmoid(self, x):
        return np.array([1/(1 + np.exp(-y)) for y in x])

    def sigmoid_deriv(self, x):
        return np.array([self.sigmoid(y)*(1-self.sigmoid(y)) for y in x])
    
    def softmax_deriv(self, x):
        return np.multiply(self.softmax(x),(1-self.softmax(x)))

    def softmax(self, x):                           # interesting overflow caused by exp
        x[x > 1e2] = 1e2
        return np.array(np.exp(x)/(np.matmul(np.ones(x.size), np.transpose(np.exp(x)))))
