# deep-learning-scripts
These are basically small projects that i used when learning deep learning for indepth understanding:

1 Loss or cost function

- The loss or cost function is a critical component in machine learning and deep learning models. It measures how well the model's predictions match the actual data. The goal of training is to minimize this function.

2  Gradient descent

-Gradient descent is a fundamental optimization algorithm used in deep learning to minimize the loss function and adjust the model's parameters (weights and biases) to achieve better performance. It is an iterative process that updates the parameters by moving them in the direction of the steepest descent, as defined by the gradient of the loss function with respect to the parameters.

3 Activation Function

- In deep learning, an activation function is a mathematical function applied to the output of a neural network layer. It determines whether a neuron should be activated or not, effectively introducing non-linearity into the model. This non-linearity allows the network to learn and model complex data patterns. Here are some common activation functions used in deep learning:
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

 
     
