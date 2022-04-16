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

![image](https://user-images.githubusercontent.com/69682199/163686104-e684ecde-4368-4cba-a274-73a2a9712010.png)

The sigmoid function is a continuous function which outputs values between 0 and 1




![image](https://user-images.githubusercontent.com/69682199/163686071-52669de0-8864-4753-9d1f-1f0770ce126f.png)


Training Algorithm:

A neural network learns by updating its weights according to a learning algorithm that helps it converge to the expected output.
Initialize the weights and biases randomly.
Iterate over the data
Compute the predicted output using the sigmoid function
Compute the loss using the square error loss function
W(new) = W(old) — α.∆W
b(new) = b(old) — α.∆b
Repeat until the error is minimal


Choosing the number of epochs and the value of the learning rate decides two things: how accurate the model is, and how fast did the model take to compute the final output.
The output with epochs = 100000 and learning rate = 0.1 is:


