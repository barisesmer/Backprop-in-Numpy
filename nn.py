import numpy as np

from cost_functions import *
from activation_functions import *

class Layer():
    def __init__(self,model, acti_func, d_acti_func,input_dim = None, output_dim = None, first_layer=False, last_layer=False,learning_rate = 0.0075):
        self.model = model
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.learning_rate = learning_rate
        

        self.A = None
        self.Z = None
        
        self.dW = None
        self.db = None
        self.dA = None # actually gradient of output of prev layer
        self.dZ = None
        
        self.first_layer = first_layer
        self.last_layer = last_layer
        
        self.acti_func = acti_func
        self.d_acti_func = d_acti_func
        
        self.next_layer = None
        self.prev_layer = None
        
    def random_initialize(self):
        self.W = np.random.randn(self.output_dim, self.input_dim)/np.sqrt(self.input_dim)  #*np.sqrt(2/self.input_dim)
        #self.W = np.random.randn(self.output_dim, self.input_dim)*0.01
        self.b = np.zeros(shape=(self.output_dim, 1))
        
    def forward_propagate(self):
        if self.first_layer:
            prev_A = self.model.data
        else:
            prev_A = self.prev_layer.A

        self.Z = self.W.dot(prev_A) + self.b
                
        self.A = self.acti_func(self.Z)
        
    def backward_propagate(self):
        if self.first_layer:
            prev_A = self.model.data
        else:
            prev_A = self.prev_layer.A
            
        if self.last_layer:
            next_dA = self.model.calculate_cost_derivative(self.A)
        else:
            next_dA = self.next_layer.dA

        m = prev_A.shape[1]
                
        self.dZ = next_dA*self.d_acti_func(self.Z)
        self.dW = self.dZ.dot(prev_A.T)/m
        self.db = np.sum(self.dZ, axis=1, keepdims=True)/m
        self.dA = self.W.T.dot(self.dZ)
        
    def optimize(self):
        self.W -= self.learning_rate*self.dW
        self.b -= self.learning_rate*self.db


#description = [{"layer_size" : 10, "activation" : "sigmoid"}, 
#               {"layer_size" : 20, "activation" : "sigmoid"}, 
#               {"layer_size" : 20, "activation" : "sigmoid"},
#               {"layer_size" : 1, "activation" : "sigmoid"}]

class NN():
    def __init__(self, description, input_size, cost_function, train_data = None, train_labels = None,learning_rate = 0.0075):
        self.learning_rate = learning_rate
        self.layers = self.create_architecture(description, input_size)
        self.data = train_data
        self.labels = train_labels
        
        # cost_function(y, y_hat)
        self.cost_function, self.d_cost_function = cost_functions[cost_function]
        
    def calculate_cost(self, y_hat):
        return self.cost_function(self.labels, y_hat)
    
    def calculate_cost_derivative(self, y_hat):
        return self.d_cost_function(self.labels, y_hat)
        
    def calculate_accuracy(self,test_data, test_labels):
        # Works for binary input right now
        self.data = test_data
        self.labels = test_labels
        
        self.forward_pass()
        
        y_hat = self.layers[-1].A
        
        pred = np.where(y_hat > 0.5, 1, 0)
            
        return (pred == self.labels).mean()
                
        
    def create_architecture(self, description, input_size):
        layers = []
        
        for index, descr in enumerate(description):
            print(index)
            input_dim = input_size if index == 0 else layers[-1].output_dim
            output_dim = descr["layer_size"]
            activ, d_activ = activation_functions[descr["activation"]]
            
            layer = Layer(self, activ, d_activ,input_dim, output_dim,
                         first_layer=(index ==  0 ), last_layer = (index == len(description) - 1),learning_rate = self.learning_rate)
            
            # set pointers
            if index != 0:
                layers[-1].next_layer = layer
                layer.prev_layer = layers[-1]
                
            layers.append(layer)
    
        # "layers" is populated now. Initialize weights
        for layer in layers:
            layer.random_initialize()
            
        return layers
    
    def add_data(self, train_data, train_labels):
        self.data = train_data
        self.labels = train_labels
        
    def forward_pass(self):
        for layer in self.layers:            
            layer.forward_propagate()
            
    def backward_pass(self):
        for layer in reversed(self.layers):
            layer.backward_propagate()
            
    def optimize(self):
        for layer in self.layers:
            layer.optimize()
    
    def train(self, epocs):
        history = []
        
        for i in range(epocs):
            self.forward_pass()
        
            cost = self.calculate_cost(self.layers[-1].A)
            history.append(cost)
            
            if i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))        
            self.backward_pass()
                        
            self.optimize()

        
        # Training done. Return history
        return history