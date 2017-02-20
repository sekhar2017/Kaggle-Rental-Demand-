import numpy as np
import pandas as pd

class NeuralNetwork(object):
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                       (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate
        self.activation_function=lambda x: 1/(1+np.exp(-x))
#         self.input_adder =np.zeros(self.weights_input_to_hidden.shape)
#         self.hidden_adder =np.zeros(self.weights_hidden_to_output.shape)
    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs =  np.dot(self.weights_input_to_hidden, inputs)
        hidden_final_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_to_output,hidden_final_outputs)
        final_outputs = final_inputs    

        ### Backward pass ###

        output_errors = ( targets - final_outputs )  # Output layer error is the difference between desired target and actual output.

        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)
        hidden_grad =  hidden_final_outputs * (1 - hidden_final_outputs)

        # TODO: Update the weights
        self.weights_hidden_to_output += self.lr *np.dot(output_errors*1,hidden_final_outputs.T)#/self.hidden_nodes  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden  += self.lr *np.dot((hidden_errors*hidden_grad),inputs.T)#/self.input_nodes # update input-to-hidden weights with gradient descent step

    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs =  np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # TODO: Output layer
        final_inputs =  np.dot(self.weights_hidden_to_output,hidden_outputs)# signals into final output layer
        final_outputs = final_inputs# signals from final output layer 
        
        return final_outputs

def MSE(y, Y):
    return np.mean((y-Y)**2)
