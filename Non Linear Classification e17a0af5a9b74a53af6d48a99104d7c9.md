# Non Linear Classification

Non-linear classification in machine learning refers to the ability of a model to classify data points into different classes based on non-linear decision boundaries. Linear classifiers, such as logistic regression and linear discriminant analysis, use a linear decision boundary to separate the data into different classes. However, in many cases, the data may not be linearly separable, and a non-linear decision boundary may be required.

# Decision Trees

Decision Trees are a type of supervised machine learning algorithm that can be used for both classification and regression tasks. They are a non-parametric method, which means they do not make assumptions about the underlying data distribution.

The basic idea behind decision trees is to recursively partition the data into subsets based on the values of the input features, with the goal of creating subsets that are as pure as possible with respect to the target variable. This process is repeated until a stopping criterion is met, such as a maximum tree depth or a minimum number of samples in a leaf node.

The decision tree is constructed by selecting the feature and the corresponding value that results in the best partition of the data. There are different metrics that can be used to evaluate the quality of a partition, such as the Gini impurity or the information gain.

The final output of a decision tree is a set of rules that can be used to make predictions for new data points. The rules are represented as a tree structure, with the internal nodes representing the features, and the leaves representing the predicted class labels.

![Untitled](Non%20Linear%20Classification%20e17a0af5a9b74a53af6d48a99104d7c9/Untitled.png)

![Untitled](Non%20Linear%20Classification%20e17a0af5a9b74a53af6d48a99104d7c9/Untitled%201.png)

Measuring impurity of a node in a decision tree is a way to evaluate the quality of a split. Impurity measures the degree to which a node's observations belong to more than one class. The goal is to create nodes with as little impurity as possible, as this indicates that the observations within that node are more similar to one another.

Two commonly used impurity measures are Gini impurity and information gain.

1. Gini impurity: It measures the probability of a randomly chosen element being incorrectly labeled if it is randomly labeled according to the class distribution in the subset. Gini impurity is calculated as the sum of the squared probabilities of each class in the node, with a value between 0 and 1. A node with a low Gini impurity (close to 0) indicates that the observations in that node are mostly of the same class.
2. Information gain: It measures the reduction in entropy that results from a split. Entropy is a measure of impurity, where a high entropy indicates a high degree of impurity. Information gain is calculated by subtracting the weighted entropy of the children nodes from the entropy of the parent node. A large information gain indicates that the split has led to a significant reduction in impurity.

![Untitled](Non%20Linear%20Classification%20e17a0af5a9b74a53af6d48a99104d7c9/Untitled%202.png)

![Untitled](Non%20Linear%20Classification%20e17a0af5a9b74a53af6d48a99104d7c9/Untitled%203.png)

# Entropy

Entropy is a measure of impurity or disorder in a dataset, often used in information theory and decision tree algorithms. It is a way to quantify the amount of uncertainty or randomness in a set of data. The entropy of a random variable X is defined as the expected value of the negative logarithm of the probability of X.

The entropy of a dataset with N classes is defined as:

H(X) = -∑(p(x) log(p(x))

Where p(x) is the probability of observing class x, and the sum is over all classes x. The entropy ranges between 0 and log2(N), with 0 representing a dataset where all observations belong to the same class (low disorder, low uncertainty), and log2(N) representing a dataset where the observations are evenly distributed among N classes (high disorder, high uncertainty).

Entropy is used in decision tree algorithms to evaluate the quality of a split. The idea is to find the feature that results in the maximum decrease in entropy, called information gain. This way of measuring impurity is based on the idea that a pure subset of data should have low entropy.

# Gain Ratio

Gain ratio is a splitting criterion used in decision tree algorithms to evaluate the quality of a split. It is an extension of the information gain criterion and is used to address the problem of bias towards features with many outcomes.

Information gain is calculated as the difference in entropy before and after a split, while gain ratio is calculated by normalizing the information gain by the intrinsic information of the feature. The intrinsic information of a feature is the average information provided by one outcome of the feature.

The gain ratio is calculated as:

GainRatio(A) = Gain(A) / SplitInformation(A)

Where A is the feature used for the split, and Gain(A) is the information gain of the feature. The SplitInformation(A) is the intrinsic information of the feature, which is calculated as:

SplitInformation(A) = -∑(p(v) log2 p(v))

Where p(v) is the probability of an outcome v of feature A, and the sum is over all outcomes v of feature A.

Gain ratio helps to address the problem of bias towards features with many outcomes by normalizing the information gain with the intrinsic information of the feature. This way, a feature with many outcomes but low intrinsic information will not be favored over a feature with fewer outcomes but high intrinsic information.

It s a ratio of InormationGain and IntrinsicValue.

![Untitled](Non%20Linear%20Classification%20e17a0af5a9b74a53af6d48a99104d7c9/Untitled%204.png)

Intrinsic value is a term used in decision tree algorithms, it is a measure of the amount of information provided by one outcome of a feature. It's the average information provided by one outcome of a feature, it's calculated as the sum of the negative logarithm of the probability of each outcome of the feature. It's used in the gain ratio splitting criterion to normalize the information gain by the intrinsic information of the feature.

In other fields, intrinsic value is also used in finance to refer to the true or underlying value of an asset, separate from its market price. In this context, intrinsic value can be calculated using various methods such as discounted cash flow analysis, or by comparing the current market price to the company's book value.

![Untitled](Non%20Linear%20Classification%20e17a0af5a9b74a53af6d48a99104d7c9/Untitled%205.png)

Pruning is a technique used in decision tree algorithms to reduce the complexity of the tree and prevent overfitting. Overfitting occurs when a model is too complex and fits the noise in the training data, rather than the underlying pattern. Pruning involves removing branches from the tree that do not contribute to the accuracy of the model.

There are different ways to perform pruning, some of the most common are:

1. Reduced Error Pruning: This is a simple and efficient method for pruning decision trees. It starts at the leaves of the tree and works backwards, removing branches that do not improve the accuracy of the model on a separate validation dataset.
2. Cost Complexity Pruning: This method involves introducing a complexity parameter, also known as cost-complexity parameter, that trades off the accuracy of the model against its complexity. The goal is to find the optimal value of the complexity parameter that minimizes the number of errors while keeping the tree as small as possible.
3. Minimum Description Length(MDL) principle: This method is based on the idea of Occam's razor, which states that among the models that fit the data, the simplest one is the best. It uses the principle of MDL to find the smallest tree that can describe the data.
4. Early stopping: This approach is used during the training process, it's a way to stop the tree from growing when the performance on a validation set starts to decrease.

![Untitled](Non%20Linear%20Classification%20e17a0af5a9b74a53af6d48a99104d7c9/Untitled%206.png)

![Untitled](Non%20Linear%20Classification%20e17a0af5a9b74a53af6d48a99104d7c9/Untitled%207.png)

# Random Forests

Random forests are an ensemble learning method for classification and regression problems. They are built by creating multiple decision trees and combining their predictions. The idea behind random forests is to leverage the power of many decision trees to improve the accuracy and stability of the model.

The process of creating a random forest involves the following steps:

1. Select random subsets of the training data and build a decision tree for each subset. This process is called bootstrap aggregating, or bagging.
2. At each node of each tree, instead of considering all features, a random subset of features is chosen as candidates for the best split. This process is called random subspace method or feature bagging.
3. Combine the predictions of all trees by taking a majority vote for classification and averaging for regression.

![Untitled](Non%20Linear%20Classification%20e17a0af5a9b74a53af6d48a99104d7c9/Untitled%208.png)

![Untitled](Non%20Linear%20Classification%20e17a0af5a9b74a53af6d48a99104d7c9/Untitled%209.png)

![Untitled](Non%20Linear%20Classification%20e17a0af5a9b74a53af6d48a99104d7c9/Untitled%2010.png)

# Multi Layer Perceptrons

![Untitled](Non%20Linear%20Classification%20e17a0af5a9b74a53af6d48a99104d7c9/Untitled%2011.png)

Perceptron training using backpropagation and gradient descent is a more advanced method for training a perceptron. It is used to train multi-layer perceptrons (MLPs) which are a type of feedforward artificial neural network that is composed of multiple layers of artificial neurons.

Backpropagation is a supervised learning algorithm that is used to train MLPs. It is used to calculate the error at the output layer and then propagate it back through the network to adjust the weights of the perceptrons in each layer.

Gradient descent is an optimization algorithm that is used to minimize the cost function that measures the difference between the predicted and true output. It is used to adjust the weights of the perceptrons in each layer so that the error is minimized.

The training process of an MLP using backpropagation and gradient descent involves the following steps:

1. Initialize the weights: The weights are initialized to small random values.
2. Feedforward: The input data is passed through the MLP, and the output is calculated using the current weights.
3. Error calculation: The difference between the predicted output and the true output is calculated. This difference is called the error.
4. Backpropagation: The error is propagated back through the network to adjust the weights of the perceptrons in each layer.
5. Gradient descent: The weights are adjusted using gradient descent so that the error is minimized.
6. Repeat steps 2-5 for a number of iterations or until the error is below a certain threshold

**Feed-forward**

The **feed-forward** pass goes as follows:

- Initialize weights with a random value
- Push input through the MLP row by row
- Calculate and push forward the activations
- Produce output value

A Multi-layer Perceptron (MLP) is a type of feedforward artificial neural network that is composed of multiple layers of artificial neurons. It is a supervised learning algorithm that is used for both classification and regression problems.

An MLP consists of an input layer, one or more hidden layers, and an output layer. The input layer receives the input data, the hidden layers process the data, and the output layer produces the final output. Each layer is composed of multiple artificial neurons, which are also called perceptrons.

The basic building block of an MLP is the artificial neuron, which is inspired by the structure and function of biological neurons. An artificial neuron receives input, performs a computation, and produces an output. The computation is performed by applying a non-linear activation function to the weighted sum of the inputs.

The training process of an MLP involves adjusting the weights of the connections between the neurons to minimize the difference between the predicted output and the true output. This is done using an optimization algorithm, such as gradient descent, which is used to minimize the cost function that measures the difference between the predicted and true output.

MLPs are widely used in a variety of applications, such as image recognition, speech recognition, and natural language processing. They can be applied to both linear and non-linear problems, and are particularly useful for problems that involve a large number of inputs or outputs.

![Untitled](Non%20Linear%20Classification%20e17a0af5a9b74a53af6d48a99104d7c9/Untitled%2012.png)

**Backpropagation**

The **backpropagation** pass goes as follows:

- Output value is compared to the label/target (this is output is first calculated using feed-forward)
- Calculate error using a loss function
- Error is propagated back through the network, layer by layer:
- Update weights depending on how much they contributed to the error, thus trying to find the weights that minimize the loss function:
- Calculate the gradients of the error function with respect to each weight
- The gradient vector indicates the direction of the highest increase in a function, while we want the highest decrease