>                                         XOR Gate using Neural Network Backpropagation Algorithm

An XOR (exclusive OR gate) is a digital logic gate that gives a true output only when both its inputs differ from each other. 
The truth table for an XOR gate is shown below:


                              X1    X2    T
                              0     0     0
                              0     1     1
                              1     0     1 
                              1     1     0

The goal of the neural network is to classify the input patterns according to the above truth table. 
If the input patterns are plotted according to their outputs, it is seen that these points are not linearly separable. 

To implement an XOR gate, I will be using a Sigmoid Neuron as nodes in the neural network. The characteristics of a Sigmoid Neuron are:
Can accept real values as input.
The value of the activation is equal to the weighted sum of its inputs i.e. ∑wi.xi
The output of the sigmoid neuron is a function of the sigmoid function, which is also known as a logistic regression function. 

![image](https://user-images.githubusercontent.com/69682199/163686192-2f54ffaa-f494-47c7-878e-745f858e2021.png)


The sigmoid function is a continuous function which outputs values between 0 and 1




![image](https://user-images.githubusercontent.com/69682199/163686071-52669de0-8864-4753-9d1f-1f0770ce126f.png)


Neural Network Diagram:

![image](https://user-images.githubusercontent.com/69682199/163686318-b44a4150-6b69-49f6-8966-25336b1f202e.png)


Training Algorithm:

A neural network learns by updating its weights according to a learning algorithm that helps it converge to the expected output.
Step1: Import the required Python libraries 
Step2: Define Activation Function : Sigmoid Function 
Step3: Initialize neural network parameters (weights, bias) 
and define model hyperparameters (number of iterations, learning rate) 
Step4: Forward Propagation 
Step5: Backward Propagation 
Step6: Update weight and bias parameters 
Step7: Train the learning model 
Step8: Plot Loss value vs Epoch 
Step9: Test the model performance 


Choosing the number of epochs and the value of the learning rate decides two things: how accurate the model is, and how fast did the model take to compute the final output.
The output with epochs = 100000 and learning rate = 0.1.


Hence, the neural network has converged to the expected output:
[0 1 1 0]. The epoch vs error graph shows how the error is minimized.

![image](https://user-images.githubusercontent.com/69682199/163686406-e0ac4c56-50ae-42bf-a19c-c040cc2c106b.png)




