# Linear Classifiers

Linear classifiers are a type of machine learning algorithm that try to find a linear boundary or a linear decision surface in order to separate the data into different classes. The most common linear classifiers are:

1. Logistic Regression: It is a linear classifier that uses a logistic function to model the probability of a binary outcome. The logistic function maps the input features to a value between 0 and 1, which can be interpreted as the probability of the positive class. Logistic regression is used for binary classification problems, such as spam detection.
2. Linear Discriminant Analysis (LDA): It is a linear classifier that tries to find a linear combination of the features that maximizes the separation between the different classes. LDA is used for multiclass classification problems, such as image classification.
3. Support Vector Machines (SVMs): They are linear classifiers that try to find the maximum margin hyperplane that separates the data into different classes. The maximum margin hyperplane is the hyperplane that is as far as possible from the closest data points of each class, called support vectors. SVMs can be used for both binary and multiclass classification problems.
4. Perceptron: It is a linear classifier that uses a linear function to model the decision boundary, it is a simple algorithm that can be used for binary classification problems.

![Untitled](Linear%20Classifiers%207e505df2237844619f8c9e92a67ff886/Untitled.png)

![Untitled](Linear%20Classifiers%207e505df2237844619f8c9e92a67ff886/Untitled%201.png)

# Cost function

The goal of the learning process is to come up with a good weight vector w. The learning process will focus on this.

# Linear regression

### Simply:

Linear regression is a statistical method for modeling the relationship between a dependent variable and one or more independent variables. It is a type of supervised learning algorithm that is used to predict a continuous target variable based on one or more input features. The goal of linear regression is to find the best linear relationship between the input features and the target variable.

Linear regression model assumes that there is a linear relationship between the input features and the target variable, which can be represented by an equation of the form:

y = b0 + b1*x1 + b2*x2 + ... + bn*xn

where y is the target variable, x1, x2, ..., xn are the input features, and b0, b1, b2, ..., bn are the coefficients of the model. The goal is to find the values of these coefficients that minimize the difference between the predicted values and the actual values of the target variable.

![Untitled](Linear%20Classifiers%207e505df2237844619f8c9e92a67ff886/Untitled%202.png)

# Gradient descent

### Batch gradient descent

Batch gradient descent is a variant of the gradient descent optimization algorithm that uses the entire dataset to compute the gradients at each iteration. It is also known as "vanilla" gradient descent.

The basic steps of batch gradient descent are:

1. Initialize the parameters with some random values or pre-specified values.
2. Compute the gradient of the cost function with respect to the parameters using the entire dataset.
3. Update the parameters by subtracting the gradient times a learning rate.
4. Repeat steps 2 and 3 for a pre-determined number of iterations or until the cost function reaches a minimum.

### Stochastic gradient descent

tochastic Gradient Descent (SGD) is a variant of the gradient descent optimization algorithm that uses only one example from the dataset to compute the gradients at each iteration. It is a computationally efficient optimization algorithm that is well-suited for large datasets and online learning.

The basic steps of Stochastic Gradient Descent are:

1. Initialize the parameters with some random values or pre-specified values.
2. Shuffle the data.
3. For each example in the dataset:
    - Compute the gradient of the cost function with respect to the parameters using only the current example.
    - Update the parameters by subtracting the gradient times a learning rate.
4. Repeat steps 2 and 3 for a pre-determined number of iterations or until the cost function reaches a minimum.

→ But, stochastic gradient descent might not converge to the local minimum, instead it will be close to it.

![Untitled](Linear%20Classifiers%207e505df2237844619f8c9e92a67ff886/Untitled%203.png)

![Untitled](Linear%20Classifiers%207e505df2237844619f8c9e92a67ff886/Untitled%204.png)

![Untitled](Linear%20Classifiers%207e505df2237844619f8c9e92a67ff886/Untitled%205.png)

# Logistic regression

Logistic Regression is a statistical method for modeling the relationship between a dependent variable and one or more independent variables. It is a type of supervised learning algorithm that is used for binary classification problems, such as spam detection, medical diagnosis, and credit risk assessment.

The basic idea behind logistic regression is to model the probability that a given input belongs to a certain class. Logistic regression uses a logistic function (also known as the sigmoid function) to map the input features to a value between 0 and 1, which can be interpreted as the probability of the positive class. The goal of logistic regression is to find the values of the parameters that maximize the likelihood of the observed data given the model.

The logistic function is defined as:

P(y=1|x;w) = 1 / (1+e^(-w*x))

where y is the binary target variable, x is the input feature and w are the parameters of the model.

The model is trained using a method called maximum likelihood estimation (MLE) which is a method of finding the values of the parameters that maximize the likelihood of the observed data given the model. Once the model is trained, it can be used to predict the probability of a new input belonging to a certain class.

## Linear vs logistic regression

Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables, where the goal is to find the best linear relationship between the input features and the target variable. It is used for predicting a continuous target variable such as the price of a house, the weight of an object, etc. Linear regression assumes that the relationship between the input features and the target variable is linear and that the target variable follows a normal distribution.

On the other hand, Logistic regression is used for binary classification problems, where the goal is to predict a binary target variable such as whether an email is spam or not, whether a person will default on a loan or not, etc. Logistic regression uses a logistic function (also known as the sigmoid function) to map the input features to a value between 0 and 1, which can be interpreted as the probability of the positive class.

## MLE

Logistic / linear regression finds the best parameters by using a method called maximum likelihood estimation (MLE). MLE is a method of finding the values of the parameters that maximize the likelihood of the observed data given the model.

The likelihood function is a measure of how well the model fits the data. It is calculated by multiplying the probability of the observed data given the model, for all the examples in the dataset. In logistic regression, the likelihood function is the product of the probability of the positive class for all the examples where the target variable is 1, and the probability of the negative class for all the examples where the target variable is 0.

To maximize the likelihood function, the algorithm needs to find the values of the parameters that maximize the probability of the observed data. This can be done by using optimization algorithms such as gradient descent, Newton-Raphson or Newton-Conjugate-Gradient.

![Untitled](Linear%20Classifiers%207e505df2237844619f8c9e92a67ff886/Untitled%206.png)

![Untitled](Linear%20Classifiers%207e505df2237844619f8c9e92a67ff886/Untitled%207.png)

![Untitled](Linear%20Classifiers%207e505df2237844619f8c9e92a67ff886/Untitled%208.png)

# Multi class classifiers

![Untitled](Linear%20Classifiers%207e505df2237844619f8c9e92a67ff886/Untitled%209.png)

n a multi-class classification problem, the goal is to predict one of multiple possible classes or labels for a given input. There are several ways to approach multi-class classification, depending on the specific problem and the available data.

![Untitled](Linear%20Classifiers%207e505df2237844619f8c9e92a67ff886/Untitled%2010.png)

# Support vector machines

Support Vector Machine (SVM) is a powerful and widely used supervised learning algorithm for both classification and regression problems. The goal of an SVM is to find the best decision boundary (also known as hyperplane) that separates the data into different classes.

SVMs are based on the idea of finding a hyperplane that maximizes the margin, which is the distance between the decision boundary and the closest data points from each class. These closest data points are called support vectors, and the decision boundary is chosen such that it is equidistant from the support vectors of each class, giving the maximum possible margin.

SVMs can be used for both binary and multi-class classification problems. For binary classification, a single SVM is trained to separate the data into two classes, while for multi-class classification, multiple binary SVMs are trained using techniques such as one-vs-one or one-vs-the-rest.

![Untitled](Linear%20Classifiers%207e505df2237844619f8c9e92a67ff886/Untitled%2011.png)

![Untitled](Linear%20Classifiers%207e505df2237844619f8c9e92a67ff886/Untitled%2012.png)

# Example in Use:

Suppose we have a dataset of two-dimensional points, and we want to classify them into two classes, "red" and "blue". The points are represented by the coordinates (x1, x2) and their class labels are represented by y, where y = 1 for "red" and y = -1 for "blue".

The goal of SVM is to find a hyperplane that separates the red points from the blue points, while maximizing the margin, which is the distance between the hyperplane and the closest data points from each class. The equation of a hyperplane is given by:

w1*x1 + w2*x2 + b = 0

Where w1 and w2 are the coefficients of the hyperplane and b is the bias term. The distance between the hyperplane and a point (x1, x2) is given by:

distance = (w1*x1 + w2*x2 + b) / sqrt(w1^2 + w2^2)

The SVM algorithm finds the best hyperplane by solving the following optimization problem:

minimize (1/2) * (w1^2 + w2^2)

subject to yi(w1*xi1 + w2*xi2 + b) >= 1 for i = 1, 2, ..., N

Where N is the number of points in the dataset. The constraints in the optimization problem ensure that the margin is greater than or equal to 1 and that all the points are on the correct side of the hyperplane.

To classify a new point (x1, x2), we can use the following equation:

y = sign(w1*x1 + w2*x2 + b)

Where sign(x) = 1 if x >= 0 and sign(x) = -1 if x < 0. If y = 1, the point is classified as "red", and if y = -1, the point is classified as "blue".