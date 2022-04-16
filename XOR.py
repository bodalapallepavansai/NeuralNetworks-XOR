'''
XOR Function using Neural Network Backpropagation Algorithm 

Architecture :

The neural network is a 3 layer network; input layer, 1 hidden layer and output layer. Input layer takes in the binary values. There are 3 nodes present in the input layer: 2 binary values and the bias term. Similarly the hidden layer contains 3 nodes : 2 values and the bias term. The output layer has 1 neuron.

Variables used :

input_layer = variable for input values 
target_output = variable that contains the proper outputs 
weights_1 = matrix of weights for the first layer 
weights_2 = matrix of weights for the second layer 
hidden_layer, output_layer = activations of the hidden layer and the final layer 
hidden_layer_error, output_layer_error = Error in the 2nd and 3rd layer 
hidden_layer_delta, output_layer_delta = Calculating the gradient between activation of a particular layer and weights associated with it.

Working :

input_layer = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]]) - Initialising the inputs for the 1st layer.

target_output = np.array([[0, 1, 1, 0]]).T

Fixing the outputs that we want out net to achieve.
weights_1 = np.random.random((3, 2)) 
weights_2 = np.random.random((3, 1))

Initialising the weight matrices for the hidden and output layer.

def sigmoid(g): 
    return 1/(1 + np.exp(-2*g))

Definition of the sigmoidal function that calculates the activation of the neurons.

def sigmoid_gradient(g): 
    return g*(1 - g)

Definition of the function that calculates the derivative of the sigmoidal function.

hidden_layer = sigmoid(np.dot(input_layer, weights_1))

Calculating the activation values by multiplying the inputs and weights associated with the input layer.

hidden_layer = hidden_layer.T hidden_layer = np.vstack((hidden_layer, bias)).T

Adding an extra row of bias values to the activation matrix.

output_layer = sigmoid(np.dot(hidden_layer, weights_2))

Calculating the activation values of the final layer by multiplying the activation values of the hidden layer and weight matrix associated with the hidden layer.

output_layer_error = target_output - output_layer 
hidden_layer_error = np.dot(output_layer_error, weights_2[0:2, :].T)*sigmoid(np.dot(input_layer, weights_1))

Calculating the error of the final layer and using that error, we calculate the error of the hidden layer.

output_layer_delta = output_layer_error * sigmoid_gradient(output_layer) 
hidden_layer_delta = hidden_layer_error * sigmoid_gradient(hidden_layer[:, 0:2])

Calculating the gradient of the error terms. This expression gives us the relation between the error terms and the neurons of a particular layer.

weights_2 += np.dot(hidden_layer.T, output_layer_delta) 
weights_1 += np.dot(input_layer.T, hidden_layer_delta)

Updating the values of weights of hidden layer and output layer.
Steps 6 through 11 are repeated for 100000 times. With every iteration, the cost reduces as the weights keep updating.



'''

import numpy as np

# Possible Inputs for XOR operation.
input_layer = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
target_output = np.array([[0, 1, 1, 0]]).T

#Random initialisation of weights
np.random.seed(1)
weights_1 = np.random.random((3, 2))
weights_2 = np.random.random((3, 1))
bias = np.ones((1, 4))

# Definition of Sigmoid Function
def sigmoid(g):
    return 1/(1 + np.exp(-g))

# Derivative of sigmoid function
def sigmoid_gradient(g):
    return g*(1 - g)


epochs = int(input(" Epochs :"))

for _ in range(epochs):
    hidden_layer = sigmoid(np.dot(input_layer, weights_1))
    hidden_layer = hidden_layer.T
    hidden_layer = np.vstack((hidden_layer, bias)).T
    output_layer = sigmoid(np.dot(hidden_layer, weights_2))
    
    # Calculating Error
    output_layer_error = target_output - output_layer
    hidden_layer_error = np.dot(output_layer_error, weights_2[0:2, :].T)*sigmoid(np.dot(input_layer, weights_1))

    # Calculating Delta Values
    output_layer_delta = output_layer_error*sigmoid_gradient(output_layer)
    hidden_layer_delta = hidden_layer_error*sigmoid_gradient(hidden_layer[:, 0:2])
    
    # Update weights
    weights_2 += np.dot(hidden_layer.T, output_layer_delta)
    weights_1 += np.dot(input_layer.T, hidden_layer_delta)


print("After training : ")
print(output_layer.flatten())
print("After Rounding off : ")
print(np.around(output_layer, decimals=0).flatten())

