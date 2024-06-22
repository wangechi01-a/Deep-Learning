# deep-learning-scripts
These are basically small projects that I used when learning deep learning for indepth understanding:

1.Loss or cost function

The loss or cost function is a critical component in machine learning and deep learning models. It measures how well the model's predictions match the actual data. The goal of training is to minimize this function.


2.Customer churn prediction - is to measure why customers are leaving a business. This challenge is based on customer churn in telecom business. The Objective is to build a deep learning model to predict the churn and use precision,recall, f1-score to measure performance of our model


3.Dropout regularization


Dropout is a regularization technique used in neural networks to prevent overfitting.Overfitting occurs when a model learns the noise in the training data rather than the actual patterns, resulting in poor performance on new, unseen data.
For each training iteration, each neuron (excluding the output neurons) has a probability 𝑝. This means the neuron is ignored in both the forward pass (calculating the output) and the backward pass (updating the weights). Typically, the dropout rate 𝑝 is set between 0.2 and 0.5.

4  Gradient descent

Gradient descent is a fundamental optimization algorithm used in deep learning to minimize the loss function and adjust the model's parameters (weights and biases) to achieve better performance. It is an iterative process that updates the parameters by moving them in the direction of the steepest descent, as defined by the gradient of the loss function with respect to the parameters.

5 Activation Function

In deep learning, an activation function is a mathematical function applied to the output of a neural network layer. It determines whether a neuron should be activated or not, effectively introducing non-linearity into the model. This non-linearity allows the network to learn and model complex data patterns. Here are some common activation functions used in deep learning:
   - Sigmoid:
     
       Range: 0 to 1
       Usage: Often used in the output layer of binary classification problems.
       Pros: Smooth gradient, outputs can be interpreted as probabilities.
       Cons: Can cause vanishing gradient problem; gradients become very small for extreme values o
     
   - Tanh
     
       Range: -1 to 1
       Usage: Preferred over sigmoid in hidden layers as it centers the data.
       Pros: Zero-centered, smooth gradient.
       Cons: Can also suffer from the vanishing gradient problem.
     
   - ReLU
     
       Range: 0 to infinity
       Usage: Most commonly used in hidden layers.
       Pros: Efficient computation, mitigates the vanishing gradient problem.
       Cons: Can suffer from the "dying ReLU" problem where neurons get stuck during training and only output zero.
     
   - Leaky ReLU
       Range: ∞ to ∞
       Usage: Helps to fix the "dying ReLU" problem.
       Pros: Allows a small gradient when 𝑥 < 0
       Cons: Requires tuning of the 𝛼 parameter

 6. Stochastic and Batch Gradient Descent
    
I used a simple home prices (banglore dataset) to implement batch and stochastic gradient descent in python. Batch gradient descent goes through  all training 
samples in forward pass to calculate cumulitive error and than we adjust weights using derivaties. In stochastic GD, we (randomly) pick one training sample, 
perform forward  pass, compute the error and immidiately adjust weights. So the key difference here is that to adjust weights batch GD will use all training 
samples where as tochastic GD will use one randomly picked training sample

8. Precision is the ratio of correctly predicted positive observations to the total predicted positives. It answers the question: Of all the instances we predicted as 
positive, how many were actually positive?
Recall (also known as Sensitivity or True Positive Rate) is the ratio of correctly predicted positive observations to all observations in the actual class. It answers 
he question: Of all the instances that are actually positive, how many did we correctly identify as positive?
The F1 Score is the harmonic mean of precision and recall. It provides a single metric that balances both concerns, useful in situations where you want to balance 
precision and recall. 


9. Mini Batch GD

It is more like SGD
Instead of choosing one (randomly picked) training sample, use a batch of randomly picked training sample
Mini batch GD: we use a batch of m samples where 0 < m < n (where n is total number of training samples)
     
