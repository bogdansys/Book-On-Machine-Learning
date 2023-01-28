# Introduction

# Introduction

The goal of machine learning is to create a model that can learn from data and can model to make predictions based on that data. This is done by training a data on a dataset and allowing it to find patterns in that data and using that data to make new prediction on new, unseen data.

# There are 2 types studied here, supervised & unsupervised ML.

## Supervised Machine Learning

Supervised learning algorithms are trained using labeled data, which means that the correct output or label for each input is provided during the training process. The goal is to learn a mapping from input variables to output variables, so that the algorithm can generalize and make predictions on new, unseen data. Common types of supervised learning include regression and classification.

## Unsupervised Machine Learning

Unsupervised learning algorithms, on the other hand, are trained using unlabeled data. The goal is to find patterns, structure, or relationships in the data, rather than making predictions. Common types of unsupervised learning include clustering and dimensionality reduction.

# Class posterior probability classifying

Class posterior probability, also known as the posterior probability of a class, is a measure of the likelihood of a given input belonging to a specific class, given the input's features. To classify an input using class posterior probability, you would calculate the posterior probability for each class, and then assign the input to the class with the highest probability.

The process of calculating class posterior probability can be broken down into the following steps:

1. Define the classes: Identify the different classes that you want to classify the input into.
2. Define the features: Identify the features of the input that will be used to make the classification.
3. Estimate class prior probabilities: Estimate the prior probability of each class based on the labeled training data.
4. Estimate class-conditional probabilities: Estimate the probability of each feature, given each class, based on the labeled training data.
5. Calculate class posterior probabilities: Use Bayes' theorem to calculate the class posterior probability for each class, given the input's features.
6. Assign the input to the class with the highest probability: The class with the highest posterior probability is the most likely class for the input.
7. Make a prediction: Once the input is classified, you can make a prediction based on the class label.

![Untitled](Introduction%20cd583f8a2ad14fd9ac47cef2d87ac82d/Untitled.png)

# Bayes’ theorem

- in many cases the posterior is hard to estimate
    - often a functional form of the class distribution can be assumed
    - we can use Bayes’ theorem to rewrite one into the other

![Untitled](Introduction%20cd583f8a2ad14fd9ac47cef2d87ac82d/Untitled%201.png)

# Bayes’ rule

1. Estimate class conditional probabilities 
2. Multiply with class priors 
3. Computer class posterior probabilities using bayes’ theorem
4. Assign object to higher probability (see → )

# Terms

### Class conditional probability

In machine learning, class-conditional probability refers to the probability of a feature, given a particular class. It is used to calculate the likelihood of a feature, given that the input belongs to a certain class. The class-conditional probability is represented as P(x|c), where x is a feature and c is a class

### Class conditional distribution

A class-conditional distribution is a probability distribution that describes the likelihood of a certain feature, given a specific class. It is a probability density function (PDF) or probability mass function (PMF) that gives the probability of observing a certain value of a feature, given a specific class.

### Class prior

A class-conditional distribution is a probability distribution that describes the likelihood of a certain feature, given a specific class. It is a probability density function (PDF) or probability mass function (PMF) that gives the probability of observing a certain value of a feature, given a specific class.

### Unconditional data distribution

In machine learning, class prior probability refers to the probability of a class occurring in the training data before taking into account the features of an input. It is also known as the prior probability of a class. The class prior probability is represented as P(c) where c is a class.

- The unconditional data distribution can be estimated from the training data using techniques such as maximum likelihood estimation (MLE) or non-parametric density estimation techniques such as kernel density estimation (KDE).
- It's important to note that the unconditional data distribution is typically used to calculate the likelihood of an input's features, regardless of the class, which is an important component in many machine learning algorithms such as Naive Bayes and Maximum likelihood classification.
- It can be used to calculate the probability of a feature occurring in the data, regardless of the class label. This can be useful in anomaly detection, where the goal is to identify instances that are unlikely under the unconditional data distribution. It can also be used to calculate the probability of an input's features under a generative model, which can then be used to generate new samples that are similar to the training data.

### Posterior probability

 In machine learning, posterior probability refers to the probability of a class, given the features of an input. It is also known as the conditional probability of a class, and is represented as P(c|x), where c is a class and x is the input.

Posterior probability is calculated using Bayes' theorem, which relates the conditional probability of an event to the reverse probability, or the prior probability of the event. The formula for Bayes' theorem is:
P(c|x) = P(x|c) * P(c) / P(x)

Where:

- P(c|x) is the posterior probability of class c given the input x
- P(x|c) is the likelihood of the input x given the class c
- P(c) is the prior probability of class c
- P(x) is the unconditional probability of the input x, regardless of the class

Once the posterior probability is calculated, it can be used to make a prediction about the class of the input. The class with the highest posterior probability is the most likely class for the input.

It's important to note that the posterior probability is based on the assumption of conditional independence of features, which might not always be the case. That's why it's important to choose the right model and features based on the problem's nature.

Bayes' theorem is a fundamental concept in machine learning, particularly in probabilistic models. It is a mathematical formula that relates the conditional probability of an event to the reverse probability, or the prior probability of the event.

The formula for Bayes' theorem is:

P(A|B) = P(B|A) * P(A) / P(B)

where:

- P(A|B) is the conditional probability of event A occurring, given that event B has occurred. This is also known as the posterior probability of A.
- P(B|A) is the conditional probability of event B occurring, given that event A has occurred. This is also known as the likelihood of B, given A.
- P(A) is the prior probability of event A occurring, regardless of whether event B has occurred.
- P(B) is the probability of event B occurring, regardless of whether event A has occurred.

In machine learning, Bayes' theorem is often used to calculate the class posterior probability, which is the probability of a given input belonging to a specific class, given its features.

For example, in a binary classification problem with two classes, C1 and C2, we can use Bayes' theorem to calculate the posterior probability of an input belonging to C1:

P(C1|X) = P(X|C1) * P(C1) / P(X)

where X is the input, P(X|C1) is the likelihood of X given that it belongs to C1, P(C1) is the prior probability of C1, and P(X) is the probability of X regardless of class.

By comparing the posterior probability of C1 with the posterior probability of C2, we can decide to which class the input belongs.

![Untitled](Introduction%20cd583f8a2ad14fd9ac47cef2d87ac82d/Untitled%202.png)

# Error

![Untitled](Introduction%20cd583f8a2ad14fd9ac47cef2d87ac82d/Untitled%203.png)

For a 2 class classification problem, 2 types of errors are defined:

- Type 1 error

![Untitled](Introduction%20cd583f8a2ad14fd9ac47cef2d87ac82d/Untitled%204.png)

- Type 2 error

![Untitled](Introduction%20cd583f8a2ad14fd9ac47cef2d87ac82d/Untitled%205.png)

If we call w1 the positive class and w2 the negative class, then e1 is false negative and e2 is false positive

# How to calculate baye’ s error

![Untitled](Introduction%20cd583f8a2ad14fd9ac47cef2d87ac82d/Untitled%206.png)

![Untitled](Introduction%20cd583f8a2ad14fd9ac47cef2d87ac82d/Untitled%207.png)

![Untitled](Introduction%20cd583f8a2ad14fd9ac47cef2d87ac82d/Untitled%208.png)

![Untitled](Introduction%20cd583f8a2ad14fd9ac47cef2d87ac82d/Untitled%209.png)

# Classifiers based on Bayes decision theorem

# Discriminative models

Discriminative models are a type of machine learning model that focuses on modeling the decision boundary or the boundary that separates different classes. They are designed to predict the class label for a given input based on its features, without modeling the underlying probability distribution of the data. (they discriminate)

- Logistic regression
- support vector machines
- decision trees
- random forests
- neural networks

# Generative models

Generative models are a type of machine learning model that focus on modeling the underlying probability distribution of the data. They are designed to learn the underlying probability distribution of the data, in order to generate new samples that are similar to the training data.(They generate data from old data)

# Multivariate gaussian

![Untitled](Classifiers%20based%20on%20Bayes%20decision%20theorem%20091a06c9d6f7461fb1f9d7a78f8075e4/Untitled.png)

![Untitled](Classifiers%20based%20on%20Bayes%20decision%20theorem%20091a06c9d6f7461fb1f9d7a78f8075e4/Untitled%201.png)

![Untitled](Classifiers%20based%20on%20Bayes%20decision%20theorem%20091a06c9d6f7461fb1f9d7a78f8075e4/Untitled%202.png)

# Plug in Baye’s rule

 In Bayesian statistics, the "plug-in" rule, also known as the "plug-in estimate" or "maximum a posteriori estimate," refers to a method of estimating the parameters of a model by plugging in the observed data into the prior distributions of the parameters.

The "plug-in" rule is a way to calculate the posterior distribution of the parameters, given the observed data and the prior distribution of the parameters. The idea is to use the maximum a posteriori estimate (MAP) of the parameters, which is the value of the parameters that maximizes the posterior distribution.

![Untitled](Classifiers%20based%20on%20Bayes%20decision%20theorem%20091a06c9d6f7461fb1f9d7a78f8075e4/Untitled%203.png)

### Estimating class priors

Just count them

![Untitled](Classifiers%20based%20on%20Bayes%20decision%20theorem%20091a06c9d6f7461fb1f9d7a78f8075e4/Untitled%204.png)

### How to estimate unconditional probabilities?

![Untitled](Classifiers%20based%20on%20Bayes%20decision%20theorem%20091a06c9d6f7461fb1f9d7a78f8075e4/Untitled%205.png)

### Estimate the class conditional probability

![Untitled](Classifiers%20based%20on%20Bayes%20decision%20theorem%20091a06c9d6f7461fb1f9d7a78f8075e4/Untitled%206.png)

# QDA - Quadratic discriminant analysis

QDA stands for Quadratic Discriminant Analysis, it is a linear classifier that is similar to Linear Discriminant Analysis (LDA), but it allows for non-linear decision boundaries.

The basic idea behind QDA is to model the class-conditional densities of the features as a multivariate Gaussian distribution for each class. The QDA algorithm estimates the mean vector and covariance matrix for each class based on the training data, and uses these estimates to calculate the likelihood of a feature vector belonging to each class. The class with the highest likelihood is chosen as the predicted class.

The decision boundary of QDA is quadratic, which means it can model non-linear decision boundaries. This makes QDA more flexible than LDA, which can only model linear decision boundaries.

The algorithm for QDA can be summarized as follows:

1. Estimate the mean vector and covariance matrix for each class based on the training data.
2. Given a new feature vector x, calculate the likelihood of x belonging to each class using the estimated mean vectors and covariance matrices.
3. Choose the class with the highest likelihood as the predicted class.

![Untitled](Classifiers%20based%20on%20Bayes%20decision%20theorem%20091a06c9d6f7461fb1f9d7a78f8075e4/Untitled%207.png)

# LDA - linear discriminant analysis

Comes from assuming all classes have the same covarience matrix.

![Untitled](Classifiers%20based%20on%20Bayes%20decision%20theorem%20091a06c9d6f7461fb1f9d7a78f8075e4/Untitled%208.png)

Linear Discriminant Analysis (LDA) is a supervised learning algorithm for classification tasks. It is a linear classifier that is used to find the best linear combination of features that separates the different classes. LDA is based on the assumption that the class-conditional densities of the features are Gaussian and have the same covariance matrix.

The basic idea behind LDA is to model the class-conditional densities of the features as a multivariate Gaussian distribution for each class. The LDA algorithm estimates the mean vector and covariance matrix for each class based on the training data, and uses these estimates to calculate the likelihood of a feature vector belonging to each class. The class with the highest likelihood is chosen as the predicted class. →

The decision boundary of LDA is linear, which means it can only model linear decision boundaries. This makes LDA less flexible than Quadratic Discriminant Analysis (QDA), which can model non-linear decision boundaries.

The algorithm for LDA can be summarized as follows:

1. Estimate the mean vector and covariance matrix for each class based on the training data.
2. Assume that the covariance matrix for all classes is the same (pooled covariance matrix)
3. Given a new feature vector x, calculate the likelihood of x belonging to each class using the estimated mean vectors and pooled covariance matrix.
4. Choose the class with the highest likelihood as the predicted class.

![Untitled](Classifiers%20based%20on%20Bayes%20decision%20theorem%20091a06c9d6f7461fb1f9d7a78f8075e4/Untitled%209.png)

![Untitled](Classifiers%20based%20on%20Bayes%20decision%20theorem%20091a06c9d6f7461fb1f9d7a78f8075e4/Untitled%2010.png)

# Nearest mean classifiers

The algorithm is trained by computing the mean of each class based on the training data. Once the means are calculated, a new input can be classified by computing its distance to the mean of each class and assigning it to the class with the closest mean. The distance metric used to calculate the distance between the input and the class means can be any appropriate distance metric such as Euclidean distance, Mahalanobis distance, etc.

The algorithm can be summarized as follows:

1. Compute the mean of each class based on the training data.
2. Given a new input x, compute the distance between x and the mean of each class
3. Assign the input to the class whose mean is closest to x.

![Untitled](Classifiers%20based%20on%20Bayes%20decision%20theorem%20091a06c9d6f7461fb1f9d7a78f8075e4/Untitled%2011.png)

# Non Parametric density estimation

Non-parametric density estimation is a method of estimating the probability density function (PDF) of a random variable without assuming a specific parametric form for the PDF. Non-parametric density estimation methods are useful when the underlying data distribution is unknown or complex, and it's difficult to assume a specific parametric form for the PDF.

Some examples of non-parametric density estimation methods include:

- Kernel Density Estimation (KDE): This method estimates the PDF by placing a kernel function (such as a Gaussian function) at each data point and summing the contributions of all the kernels. The kernel function is typically chosen to be a smooth function that is centered at each data point.
- Histograms: This method estimates the PDF by dividing the range of the data into bins and counting the number of data points in each bin. The histogram is then normalized to estimate the PDF.
- Nearest Neighbors: This method estimates the density of a point by averaging the inverse of the distance to the k-nearest points in the data set.
- Parzen Windows: This method estimates the density of a point by averaging the values of a kernel function placed at the point, over a window of fixed size.

# Histogram

Non-parametric density estimation using histograms is a method of estimating the probability density function (PDF) of a random variable without assuming a specific parametric form for the PDF. The basic idea is to divide the range of the data into a set of bins and count the number of data points that fall into each bin. The resulting histogram can be used to estimate the underlying PDF of the data.

The steps to perform non-parametric density estimation using histograms are:

1. Divide the range of the data into a set of bins. The number of bins and the width of each bin can be chosen using methods such as the "sturges rule" or the "Scott's rule".
2. Count the number of data points that fall into each bin.
3. Normalize the histogram by dividing the number of data points in each bin by the total number of data points and the width of each bin. This will give you an estimate of the PDF.
4. Plot the histogram, the resulting plot will give you a rough idea of the underlying probability density function of the data.

It's worth noting that the choice of the number of bins and the width of each bin can have a significant effect on the appearance of the histogram, using too few bins can result in a loss of information, while using too many bins can make the histogram difficult to interpret. Also, it's important to use a large enough sample to get a good estimate of the underlying density function.

# Parzen Window

Parzen window density estimation is a non-parametric method of estimating the probability density function (PDF) of a random variable. The basic idea is to place a window function (also known as a kernel function) at each data point, and sum the contributions of all the windows to estimate the PDF.

The steps to perform Parzen window density estimation are:

1. Choose a window function, such as a Gaussian or a uniform function.
2. For each data point xi, calculate the value of the window function centered at xi.
3. Sum the values of the window functions for all data points, and divide the result by the total number of data points and the volume of the window function. This will give you an estimate of the PDF.
4. Plot the resulting estimate, the plot will give you a rough idea of the underlying probability density function of the data.

![Untitled](Non%20Parametric%20density%20estimation%20aa28900501a54df6bda8237c0f6cbeb7/Untitled.png)

## Example:

Let's say we have a dataset of 1000 two-dimensional data points, and we want to estimate the probability density function (PDF) of the data using the Parzen window density estimation method.

1. Choose a window function: We can choose a Gaussian window function with a standard deviation of 0.5.
2. For each data point, calculate the value of the window function centered at the data point: For each data point xi, we calculate the value of the Gaussian function at xi, using the formula for a 2-dimensional Gaussian:

g(x;μ,Σ) = (1/(2π)^(k/2) * |Σ|^(1/2)) * exp( -1/2 * (x-μ)^T * Σ^-1 * (x-μ) )

Where x is the data point, μ is the mean of the window function (which is xi in this case), Σ is the covariance matrix of the window function (which is 0.25*I in this case), and k is the number of dimensions of the data (which is 2 in this case).

1. Sum the values of the window functions for all data points, and divide the result by the total number of data points and the volume of the window function: We sum the values of the window functions for all data points, divide the result by 1000 (the total number of data points) and the volume of the window function (which is (2π*0.25)^(1/2) in this case). This will give us an estimate of the PDF.
2. Plot the resulting estimate: We can plot the resulting estimate using a heat map or a contour plot. The plot will give us a rough idea of the underlying probability density function of the data.

### Reminder PDF:

A probability density function (PDF) is a mathematical function that describes the probability of a continuous random variable taking on a particular value. The PDF is a function that describes the relative likelihood of a random variable taking on a particular value, and it is a useful tool for understanding and characterizing the behavior of a continuous random variable.

The PDF is defined for a continuous random variable X with the following properties:

- It is non-negative: f(x) ≥ 0 for all x in the domain of X
- The total area under the curve of the PDF is equal to 1, this is known as "probability integral"
- The probability that a random variable X takes on a value in a particular range is equal to the area under the PDF curve over that range

For example, a normal distribution's PDF is described by the well-known bell-shaped curve, it has a single peak and it is symmetric. The height of the curve at any point is the probability density of the variable at that point.

# K nearest neighbours

The k-nearest neighbors (KNN) algorithm is a non-parametric method for classification and regression tasks. It is a simple, yet powerful algorithm that can be used for a wide range of applications. The basic idea behind KNN is to find the k data points in the training set that are closest to a new input, and use the class labels of these k nearest neighbors to make a prediction for the new input.

The algorithm is trained on a labeled dataset, in which each data point is represented by a set of features and a class label. Once the training set is created, the algorithm can be used to classify new inputs by finding the k nearest neighbors in the training set and using the majority class label among these k nearest neighbors as the predicted class label for the new input.

### Steps:

1. Choose a value for k, this value represents the number of nearest neighbors that will be considered to make a prediction.
2. Given a new input x, calculate the distance between x and all the points in the training set, this distance can be calculated using any appropriate distance metric such as Euclidean distance, Manhattan distance, etc.
3. Find the k nearest neighbors to x by selecting the k points from the training set that are closest to x.
4. Use the class labels of the k nearest neighbors to make a prediction for x. A common approach is to choose the most common class label among the k nearest neighbors, but other approaches are possible such as weighted voting or considering the distance to each neighbor in the decision.

The KNN algorithm is sensitive to the choice of k, a large value of k will make the algorithm more robust to noise but less sensitive to small variations in the input, on the other hand, a small value of k will make the algorithm more sensitive to small variations in the input but more susceptible to noise. Additionally, the performance of the KNN algorithm can be affected by the dimensionality of the data, as the distance metric becomes less meaningful in high-dimensional spaces.

### In case of a tie:

- use odd  k
- random flip
- prior (chose class with highest prior)

### Distance measures

In the k-nearest neighbors (KNN) algorithm, the choice of distance measure can have a significant effect on the performance of the algorithm. Some commonly used distance measures in KNN are:

1. Euclidean Distance: This is the most commonly used distance measure in KNN. It calculates the straight-line distance between two points in n-dimensional space. It is defined as the square root of the sum of the squares of the differences between the coordinates of the two points.
2. Manhattan Distance: Also known as the "taxi-cab" distance, this measure calculates the distance between two points by summing the absolute differences of their coordinates. It is suitable for data that has categorical features.
3. Minkowski Distance: This distance measure is a generalization of the Euclidean and Manhattan distance, it is defined as the sum of the absolute differences of their coordinates raised to a power p (p>=1).

### So how do we select ﻿*k*﻿?

- Set aside a portion of the training data
- Vary ﻿*k*
- Pick the﻿*k* that gives the best generalization performance
    
    

# Naive Bayes’

Naive Bayes is a classification algorithm based on Bayes' theorem, which is a mathematical formula for calculating conditional probabilities. It is called "naive" because it makes a strong independence assumption between the features, meaning that it assumes that the presence or absence of a particular feature in the input has no effect on the presence or absence of any other feature.

![Untitled](Non%20Parametric%20density%20estimation%20aa28900501a54df6bda8237c0f6cbeb7/Untitled%201.png)

### Zero frequency problem

When we are classifying emails as spam or not spam, based on the words inside the email, it may occur that we see a word that has never occurred before. Stated more broadly, it’s a bad idea to estimate the probability of some event to be zero, just because we haven’t seen it before.

To solve this problem, we will never allow zero probabilities.

### Fooling Naive Bayes

We can easily fool Naive Bayes. If we were to base the decision about whether an email is a spam email on the amount of words common in spam relative to the amount of valid words, we can just add lots of valid words into our spam email, and since every word contributes independently, we won’t catch it.

### Missing Data

Suppose we don’t have a value for some attribute ﻿��*xj*﻿. This is easy to solve with Naive Bayes. We can just ignore the attribute instance where it’s missing a value, and only compute the likelihood based on observed values.

### Pros & cons of Naive Bayes

Pros:

- Can handle high dimensional feature spaces
- Fast training time
- Can handle missing values
- Transparent

Cons:

- Can’t deal with correlated features

### Example:

An example of using the Naive Bayes algorithm for a binary classification problem is spam detection. The algorithm would be trained on a labeled dataset of email messages, where each message is represented by a set of features (such as the presence or absence of certain words or phrases) and a class label (spam or not spam).

Once the training set is created, the algorithm can be used to classify new emails by calculating the probability that they are spam or not spam based on the presence or absence of certain features. The basic assumptions of the Naive Bayes algorithm are:

1. Feature independence: It assumes that the presence or absence of a particular feature in the input is independent of the presence or absence of any other feature. In the case of spam detection, this means that the presence of the word "Viagra" in an email is independent of the presence of the word "Nigeria".
2. Conditional independence: It assumes that the features in the input are conditionally independent given the class label. In the case of spam detection, this means that the presence of the word "Viagra" in an email is independent of the presence of the word "Nigeria" given that the email is spam.
3. Class-conditional independence: It assumes that the class labels are conditionally independent given the features. In the case of spam detection, this means that the probability of an email being spam is independent of the probability of another email being spam given the presence or absence of certain words or phrases.

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

# Fairness

### 

### Independence Criterion

The Independence criterion in the context of machine learning fairness is a requirement that the decisions made by a model should be independent of certain protected attributes, such as race, gender, or religion. This means that the model's predictions should not be systematically different for different groups of people based on these protected attributes.

### Separation Criterion

The Separation criterion is a stronger requirement for fairness, which requires that the model's predictions for different groups of people should be equivalent or indistinguishable. This means that the model should not only be independent of protected attributes but also that its performance should be the same across different groups.

In other words, the Independence criterion states that the model should not discriminate based on protected attributes, while the Separation criterion states that the model should not only not discriminate but also should be fair in its predictions for all groups.

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
# Empirical Risk Minimisation

![Untitled](Empirical%20Risk%20Minimisation%201ad47dae35e54609b98feec8502bc6dd/Untitled.png)

Empirical risk minimization (ERM) is a principle used in statistical learning and machine learning to minimize the generalization error of a model on unseen data. It is used to find the best parameters of a model that minimize the difference between the predicted values and the true values on the training data.

The basic idea of ERM is to define a loss function, which measures the difference between the predicted and true values, and then find the set of parameters that minimize the average value of this loss function over the training data. This can be done using optimization algorithms such as gradient descent.

ERM is widely used in supervised learning, where the goal is to find the best model that can accurately predict the target variable based on the input features. The choice of the loss function depends on the type of problem, for example, for classification problems, the cross-entropy loss function is commonly used, while for regression problems the mean squared error loss function is used.

ERM has several advantages, including its simplicity and ease of implementation. It can be used with a wide range of different models and it is easy to adapt to different types of data and different types of problems. However, it is worth noting that ERM can be sensitive to the choice of the loss function, and it can be affected by the presence of noise or outliers in the data.

In summary, Empirical Risk Minimization is a principle used in statistical learning and machine learning to minimize the generalization error of a model on unseen data, it uses a loss function to measure the difference between the predicted and true values and it finds the set of parameters that minimize the average value of this loss function over the training data. It is widely used in supervised learning and it has several advantages, but it is sensitive to the choice of the loss function and it can be affected by the presence of noise or outliers in the data.
# Unsupervised Learning

Unsupervised learning is a type of machine learning where the goal is to find patterns or structure in a dataset without using labeled data. Unlike supervised learning, where the goal is to predict an output based on input features, unsupervised learning does not have a specific target variable to predict. Instead, it aims to find underlying patterns and relationships in the data. The data is unlabeled.

- Clustering
- Dimensionality reduction

![Untitled](Unsupervised%20Learning%203b7412fa80af49cf9efa5c5b22a9d11c/Untitled.png)

←All of these are supervised ML models, meaning that they are based on labeled data. In unsupervised learning, they are not labeled.

# Dimensionality Reduction

Dimensionality reduction is a technique used to reduce the number of features (dimensions) in a dataset while preserving as much information as possible. The goal of dimensionality reduction is to simplify the data while maintaining its essential characteristics, which can improve the performance of machine learning algorithms, speed up their training process and make the data more interpretable.

Principal Component Analysis (PCA) is a technique used for dimensionality reduction in which the data is transformed into a new coordinate system where the new axes are the principal components of the data. The principal components are the directions that capture the most variation in the data.

The steps for PCA are as follows:

1. Mean normalization: The data is centered around the mean by subtracting the mean from each feature.
2. Covariance matrix: The covariance matrix of the normalized data is calculated.
3. Eigenvectors and eigenvalues: The eigenvectors and eigenvalues of the covariance matrix are calculated. The eigenvectors are the principal components, and the eigenvalues are the variance of the data in the direction of the eigenvectors.
4. Dimensionality reduction: The eigenvectors with the highest eigenvalues are chosen to form a new coordinate system. The new coordinate system captures the most variation in the data, and the number of axes can be reduced by discarding the eigenvectors with lower eigenvalues.
5. Data transformation: The original data is transformed into the new coordinate system by multiplying it with the matrix of eigenvectors.

![Untitled](Unsupervised%20Learning%203b7412fa80af49cf9efa5c5b22a9d11c/Untitled%201.png)

![Untitled](Unsupervised%20Learning%203b7412fa80af49cf9efa5c5b22a9d11c/Untitled%202.png)

## Example Covarience Matrix:

![Untitled](Unsupervised%20Learning%203b7412fa80af49cf9efa5c5b22a9d11c/Untitled%203.png)

![Untitled](Unsupervised%20Learning%203b7412fa80af49cf9efa5c5b22a9d11c/Untitled%204.png)

# Clustering

Using clustering we can find natural groups in data where:

- Items within the groups are close togethers
- Items between groups are far apart

### 

![Untitled](Unsupervised%20Learning%203b7412fa80af49cf9efa5c5b22a9d11c/Untitled%205.png)

# Cluster evaluation

We have a few measures for that:

- **Intra-cluster cohesion** (compactness), measures how near the data points are to the cluster's mean. Calculated by the sum of squared errors
- **Inter-cluster seperation** (isolation), measures the distance between two clusters, which should be as large as possible

![Untitled](Unsupervised%20Learning%203b7412fa80af49cf9efa5c5b22a9d11c/Untitled%206.png)

# Clustering Techniques

## K means clustering

![Untitled](Unsupervised%20Learning%203b7412fa80af49cf9efa5c5b22a9d11c/Untitled%207.png)

## Hierarchical Clustering

![Untitled](Unsupervised%20Learning%203b7412fa80af49cf9efa5c5b22a9d11c/Untitled%208.png)

![Untitled](Unsupervised%20Learning%203b7412fa80af49cf9efa5c5b22a9d11c/Untitled%209.png)

K-means is a centroid-based clustering algorithm that is used to group similar data points together into clusters. The goal of k-means is to minimize the within-cluster variance, which is the sum of the squared distances between each data point and the centroid of its cluster.

The steps for k-means clustering are as follows:

1. Initialization: Choose k centroids randomly from the data.
2. Assign each data point to the closest centroid: Each data point is assigned to the cluster whose centroid is closest to it.
3. Recalculate the centroids: The centroids are recalculated as the mean of the data points in the cluster.
4. Repeat steps 2 and 3 until the centroids no longer change.

![Untitled](Unsupervised%20Learning%203b7412fa80af49cf9efa5c5b22a9d11c/Untitled%2010.png)

![Untitled](Unsupervised%20Learning%203b7412fa80af49cf9efa5c5b22a9d11c/Untitled%2011.png)

![Untitled](Unsupervised%20Learning%203b7412fa80af49cf9efa5c5b22a9d11c/Untitled%2012.png)

![Untitled](Unsupervised%20Learning%203b7412fa80af49cf9efa5c5b22a9d11c/Untitled%2013.png)

# Classifier Evaluation

![Untitled](Classifier%20Evaluation%203eeacd065e3e4244addf3ae8ab32d7fe/Untitled.png)

*Based on Bayes Decision Theory*

- **Quadratic Discriminant Classifier:** a classifier yielding a quadratic curve as decision boundary, used when the covariance matrix is different between the classes
- **Linear Discriminant Classifier:** a linear classifier used when the covariance matrix is the same for all classes
- **Nearest Mean Classifier:** a linear classifier which uses the distance to the mean of each class and creates a decision boundary based on that

*Non-parametric*

- ***k*﻿-nearest neighbors:** look for the﻿*k* nearest neighbors and pick the class with the most samples in this region
- **Parzen:** place cells of fixed size around the data points and decide which class to choose based on posterior probability

*Linear*

- **Logistic:** the go-to method for binary classification, modelling the probability of the first class
- **Support Vector:** find a hyper-plane that maximizes the margin between positive and negative examples

*Non-linear*

- **Decision tree:** ask yes or no questions to split the data
- **Neural network:** multi layer perceptrons which all do binary classifications, combined give a class estimate

## Test / training set divison

![Untitled](Classifier%20Evaluation%203eeacd065e3e4244addf3ae8ab32d7fe/Untitled%201.png)

# Learning curves

A learning curve shows how our error is changing with the number of training samples. In principle, this error is the true error. So learning curves are curves that plot classification errors against the number of samples in training set, which gives insight in things like overtraining, usefulness of additional data, comparison between different classifiers etc.

![Untitled](Classifier%20Evaluation%203eeacd065e3e4244addf3ae8ab32d7fe/Untitled%202.png)

### Bootstrapping

Given a training set ﻿�*D*﻿ of size ﻿�*n*﻿, generate ﻿�*m*﻿ new training sets ﻿��*Di*﻿, each of size ﻿�′*n*′﻿ . We generate these training sets by randomly choosing elements from the entire dataset. This means it is possible that you select a certain elements twice, and train on it twice. This makes re-sampling truly random instead of choosing from the left over data set.

The classification error estimate is the average of the classification errors.

### *k*﻿-fold cross-validation

Divide the dataset in groups of ﻿�*k*﻿ samples. Use ﻿11﻿ sample for testing, ﻿�−1*k*−1﻿ for training. Each time you train and test your error, then you repeat this until you have tested on all ﻿�*k*﻿ groups.

The classification error estimate is the average of the classification errors.

### Leave-one-out cross-validation

Leave-one-out is exactly the same as ﻿�*k*﻿-fold cross-validation, where ﻿�=1*k*=1﻿. So you train ﻿�*k*﻿ times, each with only ﻿11﻿ test example.

This is optimal for training, but is very computationally intensive. That's why ﻿1010﻿-fold cross-validation is often used (﻿�=10*k*=10﻿)

### Double cross-validation

Machine learning methods have "hyperparameters", which are parameters of the machine learning method itself (like the width ﻿ℎ*h*﻿ in Parzen). You shouldn't optimize the parameters of the learning methods by looking at the test set.

You should optimize them by using cross-validation inside another cross-validation (internal cross-validation). Once you found the value of the hyperparameter with the lowest error, retrain the entire classifier with the optimal ﻿ℎ�*hj*﻿.

It is possible that the different folds yield different parameters with the same error, then you can just average them and retrain.

But, we can also look at the apparent error on the training set. The higher the amount of samples, the higher the error seems to be, but the lower the true error will be. This is because with a low amount of samples, we have a lot of overfitting. So these learning curves give plots the error on both the training and the test set.

Realistically, these learning curves have a very large variability. To fix this, we average them a few times.

# Confusion matrix

 

![Untitled](Classifier%20Evaluation%203eeacd065e3e4244addf3ae8ab32d7fe/Untitled%203.png)

As you can see, you can calculate the averaged error using this confusion matrix, by averaging the percentage that is misclassified between the classes. Note that all classes are equally important here.

If you see that classification of a certain class is very good but classification of other classes is poor, you can remove this good class from the classifier and make a new classifier for the other classes, which will probably use different features and parameters.

Another thing you can use the confusion matrix for is by seeing that classes with a low prior are classified poorly. In some cases, that should not be the case (e.g. diagnosing a disease). Then you can use this confusion matrix and a cost function in order to determine what to do.

A confusion matrix is a table that is used to evaluate the performance of a classification model. It is used to compare the predicted class labels with the true class labels in a test set. The matrix is often used to evaluate the accuracy of a classification algorithm.

A confusion matrix typically has two rows and two columns for binary classification and more rows and columns for multi-class classification problem. The rows represent the actual class labels, and the columns represent the predicted class labels. The entries in the matrix represent the number of instances that have a particular combination of actual and predicted class labels.

The entries in a confusion matrix are often represented as follows:

- True Positives (TP): The number of instances that are correctly classified as positive.
- False Positives (FP): The number of instances that are incorrectly classified as positive.
- True Negatives (TN): The number of instances that are correctly classified as negative.
- False Negatives (FN): The number of instances that are incorrectly classified as negative.

 Def! → A confusion matrix is a table that is used to evaluate the performance of a classification model. It is used to compare the predicted class labels with the true class labels in a test set. The matrix is often used to evaluate the accuracy of a classification algorithm.

A confusion matrix typically has two rows and two columns for binary classification and more rows and columns for multi-class classification problem. The rows represent the actual class labels, and the columns represent the predicted class labels. The entries in the matrix represent the number of instances that have a particular combination of actual and predicted class labels.

The entries in a confusion matrix are often represented as follows:

- True Positives (TP): The number of instances that are correctly classified as positive.
- False Positives (FP): The number of instances that are incorrectly classified as positive.
- True Negatives (TN): The number of instances that are correctly classified as negative.
- False Negatives (FN): The number of instances that are incorrectly classified as negative.

# Rejection curves

We had the assumption that the data we get in training is about similarly distributed in reality. But, sometimes the errors we make are very costly and there might be very little examples of a certain class, which makes it harder to model the distribution. So there are two types of input data we can reject.

### Outlier rejection

We reject objects that are far away from the training data. So reject if the probability that we get from our model is basically ﻿00

### Ambiguity rejection

Reject objects for which classification is unsure, i.e. when the posterior probabilities are about equal

![Untitled](Classifier%20Evaluation%203eeacd065e3e4244addf3ae8ab32d7fe/Untitled%204.png)

![Untitled](Classifier%20Evaluation%203eeacd065e3e4244addf3ae8ab32d7fe/Untitled%205.png)

# 2 class Erorrs

![Untitled](Classifier%20Evaluation%203eeacd065e3e4244addf3ae8ab32d7fe/Untitled%206.png)

Take for example:

- **Sensitivity** of a target class: performance for object from that target class
- **Specificity**: performance for all objects outside target classs

# ROC curves

![Untitled](Classifier%20Evaluation%203eeacd065e3e4244addf3ae8ab32d7fe/Untitled%207.png)

![Untitled](Classifier%20Evaluation%203eeacd065e3e4244addf3ae8ab32d7fe/Untitled%208.png)

If we were to use a logistic regression algorithm and we classify data, we will get correct classifications and false positives/negatives. We want to improve our amount of correct classifications, we could do this by changing our threshold and trying again. But this gives many confusion matrices, which is very inefficient.

So we can use the ROC curve to plot false positives vs true positives, performance vs cost etc.

→ Notes: 

ROC (Receiver Operating Characteristic) curves are a graphical representation of the trade-off between the true positive rate and the false positive rate of a classification model. The ROC curve is a plot of the true positive rate (sensitivity) against the false positive rate (1-specificity) for different thresholds.

The true positive rate (TPR) is the proportion of true positive instances (correctly classified as positive) out of the total number of positive instances. It is also known as the sensitivity of the model.

The false positive rate (FPR) is the proportion of false positive instances (incorrectly classified as positive) out of the total number of negative instances. It is also known as the fall-out of the model.

An ROC curve plots TPR against FPR for different thresholds. A perfect classifier would have a TPR of 1 and an FPR of 0, which would result in an ROC curve that hugs the top left corner of the plot. A classifier that performs no better than random guessing would result in an ROC curve that is a diagonal line from the bottom left to the top right corner of the plot.

The Area Under the ROC Curve (AUC) is a measure of the overall performance of a classifier. A perfect classifier would have an AUC of 1, while a classifier that performs no better than random guessing would have an AUC of 0.5.

# OverView

# Normal Based (based on normal distribution)

- Bayes rule
    - Bayes' rule is a theorem that states that the probability of an event occurring is based on prior knowledge of related conditions. It's represented mathematically as P(A|B) = P(B|A) * P(A) / P(B) where P(A|B) is the probability of A given B, P(B|A) is the likelihood of B given A, P(A) is the prior probability of A and P(B) is the prior probability of B.
- Multivariate Gaussian distributions
    - It is defined by its mean vector and covariance matrix and can be used for various tasks such as density estimation, clustering, and as a likelihood function in various models such as Gaussian Mixture Model, Gaussian Processes etc. It can be used in supervised and unsupervised learning to understand the underlying pattern of the data. The probability density function of the distribution is used to calculate the likelihood of the data given the model's parameters.
- Curse of dimensionality
    - The curse of dimensionality refers to the phenomenon where the amount of data required to effectively train a model increases exponentially as the number of dimensions (features) in the data increases
- Discriminant functions
    - Discriminant functions, also known as decision boundaries or classifiers, are used in machine learning to separate different classes or categories of data. They are mathematical functions that take in a set of input features and output a prediction of the class or category that the input belongs to.
        - Linear discriminant functions: These are linear equations that separate the data using a straight line or hyperplane. Examples include linear discriminant analysis (LDA) and linear regression.
        - Quadratic discriminant functions: These are quadratic equations that separate the data using a curved boundary. An example is Quadratic discriminant analysis (QDA)
        - Non-linear discriminant functions: These are more complex functions that can separate the data using non-linear boundaries. Examples include decision trees, random forests, and neural networks.
- Covariance matrix
    - A covariance matrix is a square matrix that describes the covariance between a set of random variables. It is used to represent the variability and the correlation between different variables in a dataset. The diagonal elements of a covariance matrix represent the variances of the individual variables, while the off-diagonal elements represent the covariances between the variables.
    
    The covariance matrix is defined as:
    
    C = 1/(n-1) * (X - mu)^T * (X - mu)
    
    where X is the n x d matrix of observations, mu is the n x d matrix of the mean of observations, n is the number of observations, and d is the number of variables.
    
- Quadratic classifier (each class has mean and covariance matrix)
    - A quadratic classifier is a type of discriminant function that uses a quadratic boundary to separate different classes of data. It is a non-linear classifier that can separate data using a curved boundary as opposed to a linear boundary used by linear classifiers such as linear discriminant analysis (LDA) and logistic regression.
- Linear classifier (each class has mean, all classes have the same covariance matrix)
    - A linear classifier is a type of discriminant function that separates different classes of data using a linear boundary, such as a straight line or a hyperplane. Linear classifiers are based on the assumption that the data is linearly separable, meaning that a straight line or a hyperplane can be used to separate the classes with minimal misclassification error.
    
    Examples of linear classifiers include:
    
    - Linear Discriminant Analysis (LDA)
    - Logistic Regression
    - Perceptron
    - Support Vector Machines (SVMs) with linear kernel
- Nearest mean (each class has mean, all classes have identity covariance matrix)
    - First, the algorithm is trained on a labeled dataset, in which the mean of each class is calculated.
    - Next, for a new unseen data point, the algorithm calculates the distance between the point and the means of all classes.
    - The class with the closest mean is then assigned to the new data point.
        - The nearest mean classifier is a variation of the k-nearest neighbors (k-NN) algorithm, where k=1. It is similar to the k-NN algorithm, but it only considers the mean of each class, rather than all the points in the class. It is a linear classifier because it makes predictions based on a linear combination of the input features.

# Non-parametric

- Different distance measures (Euclidean, Manhattan, hammen)
1. Euclidean distance: This is the most widely used distance measurement and it is based on the Pythagorean theorem. It calculates the straight-line distance between two points in a multidimensional space. It is defined as the square root of the sum of the squares of the differences between the coordinates of the two points.
2. Manhattan distance (also known as "taxi-cab" distance): It is calculated as the sum of the absolute differences between the coordinates of the two points. It is commonly used in cases where the data is measured on different scales and it is also used in the k-NN algorithm.
3. Hamming distance: It is a measurement of the difference between two strings. It is the number of positions at which the corresponding symbols are different. It is used for categorical data and it is commonly used in text classification and DNA sequence analysis.
- Learning curves
    
    Learning curves in machine learning are plots that show the relationship between the size of the training set and the performance of the model. These plots are used to diagnose if a model is underfitting, overfitting or has a good fit to the data.
    
    A learning curve will typically have two lines: one for the training set and one for the validation set. If a model is underfitting, the training and validation error will be high, and both lines will be close together. If a model is overfitting, the training error will be low, but the validation error will be high, and the two lines will be far apart. If a model has a good fit to the data, the two lines will converge to a low error as the size of the training set increases.
    
- Zero frequency problem
    
    The zero frequency problem, also known as the zero probability problem, is a problem that arises in natural language processing and text classification when trying to estimate the probability of a word or phrase that has not been seen in the training data.
    
    In text classification, a common approach is to use a bag-of-words representation, where each document is represented as a vector of the frequency of words in the document. The problem is that some words may not be present in the training data, which means that their frequency is zero. This can cause problems when trying to estimate the probability of these words, as dividing by zero is undefined.
    
    One common solution to this problem is to add a small constant to the frequency of each word, known as Laplace smoothing or add-k smoothing. This ensures that all words have a non-zero probability, and it also helps to avoid overfitting.
    
- Histogram (put into buckets)
    - The method works by dividing the range of the data into a set of bins and counting the number of data points in each bin. The resulting histogram gives an estimate of the probability density function of the dataset. The width of the bins and the number of bins are controlled by parameters that affect the shape of the histogram and the accuracy of the density estimate.
- Parzen / Kernel density estimation (find﻿*k* given﻿*v*﻿. Put a kernel function on each data point)
    - non-parametric method used to estimate the probability density function (PDF) of a random variable. It is a way to estimate the shape of the underlying distribution of a dataset without making any assumptions about the underlying distribution.
- *k*﻿-nearest-neighbours (find﻿*v* given﻿*k*﻿. Find the﻿*k* nearest neighbors, take most common class)
    - The k-nearest neighbors (k-NN) algorithm is a non-parametric method for classification and regression tasks in machine learning. It works by finding the k training examples that are closest to a new test example, and then classifying the test example based on the majority class among its k-nearest neighbors. The distance metric used to determine the similarity between examples can be Euclidean, Manhattan or other types of distance measure.
- Naive bayes (estimate each feature separately, multiply probabilities)
    - Naive Bayes is a probabilistic algorithm for classification that uses Bayes Theorem with strong (naive) independence assumptions between the features. It is a fast and simple algorithm that can be applied to a variety of classification problems such as text classification, sentiment analysis and spam filtering.

# Definitions

Overfitting and underfitting are common problems in machine learning that occur when a model is either too complex or too simple to accurately capture the underlying patterns in the data.

Overfitting occurs when a model is too complex and it fits the noise or random variations in the training data, instead of the underlying patterns. This means that the model will perform well on the training data but poorly on unseen data, also known as high variance. This can be caused by having too many features or parameters in the model, or by having too few training examples.

Underfitting occurs when a model is too simple and it fails to capture the underlying patterns in the data. This means that the model will perform poorly on both the training data and unseen data, also known as high bias. This can be caused by having too few features or parameters in the model, or by having too much regularization.

To avoid overfitting, one can use techniques such as reducing the number of features, regularization, and early stopping. To avoid underfitting, one can use techniques such as adding more features, increasing the complexity of the model, and using more data for training.

# Linear

- Cost functions
    - 
    
    Linear discriminant cost functions are mathematical functions used to find the linear boundary that separates different classes of data in a supervised classification task. These functions are used to minimize the classification error by adjusting the parameters of the linear boundary.
    
    Some examples of linear discriminant cost functions include:
    
    - Linear Discriminant Analysis (LDA): This function finds the linear combination of features that maximizes the ratio of the between-class variance to the within-class variance.
    - Logistic Regression: This function finds the linear boundary that maximizes the likelihood of the data given the class labels.
- Gradient descent
    - Gradient descent is an optimization algorithm used to minimize a cost function by iteratively moving in the direction of steepest decrease of the cost function. It is widely used in machine learning to train models by updating the parameters to minimize the error. The algorithm stops when the change in the cost function is below a certain threshold or when a maximum number of iterations is reached.
- Multi-class logistic classification (one-versus-the-rest and one-versus-one)
    
    Multi-class logistic classification is a method for classification tasks with more than two classes. There are two common approaches to handle multi-class classification using logistic regression:
    
    - One-versus-the-rest (OvR): also known as one-vs-all, it trains multiple binary classifiers, one for each class. Each classifier is trained to separate the current class from all the others.
    - One-versus-one (OvO): It trains multiple binary classifiers, one for each pair of classes. For example, if there are three classes A, B and C, three binary classifiers will be trained: A vs B, A vs C and B vs C.
- Hinge loss
    
    Hinge loss is a loss function used for training linear classifiers such as Support Vector Machines (SVMs) and linear perceptrons. It is particularly used for training models with a maximum-margin criterion, which aims to find a decision boundary with the largest possible margin between the two classes.
    
    The hinge loss function is defined as the maximum between 0 and the difference between the true label and the predicted label, added a margin. This means that the loss is zero when the predicted label is correct, and it increases as the predicted label deviates from the true label.
    
    Hinge loss is a convex function and it is differentiable, it can be optimized using gradient descent or other optimization algorithms. It can be also used in multi-class classification problems by using a one-vs-all or one-vs-one approach.
    
- Least mean squares
    - east Mean Squares (LMS) is a stochastic gradient descent algorithm used to find the weights of a linear model that minimize the mean squared error between the predicted output and the true output. The algorithm starts with an initial guess of the weights, and iteratively updates the weights by subtracting a small step multiplied by the gradient of the mean squared error with respect to the weights.
- Linear regression (find best hyperplane through data using gradient descent)
- Logistic classifier (linear classification using logistic function)
- Support vector machine (try to linearly separate a space)

# Responsible machine learning

- Bias, prejudice and discrimination
- Implicit vs explicit bias
- Fairness in ML

# Non-linear

- Combining classifiers (soft-type and hard-type)
- Decision trees (Create a tree by repeatedly splitting axis, using entropy)
- Multi-layer perceptrons (Create multiple layers of linear classifiers)

# Unsupervised

- Principal component analysis
- Clustering (find groups of data that are close together)
- *k*﻿-means clustering
- Hierarchical clustering
- Bottom up vs top down

# Evaluation

- Error estimation
- Training set, test set and validation set
- Bootstrapping
- *k*﻿-fold cross-validation
- Learning curves
- Feature curves
- Bias-variance dilemma
- Confusion matrices
- Rejection curves
- ROC curves

# Examples

# Computing a ROC curve

![Untitled](Examples%207c0ce7a52f834d068963e03e650ac3a3/Untitled.png)

![Untitled](Examples%207c0ce7a52f834d068963e03e650ac3a3/Untitled%201.png)

# Hierarchical Clustering

![Untitled](Examples%207c0ce7a52f834d068963e03e650ac3a3/Untitled%202.png)

![Untitled](Examples%207c0ce7a52f834d068963e03e650ac3a3/Untitled%203.png)

# PCA component analysis

![Untitled](Examples%207c0ce7a52f834d068963e03e650ac3a3/Untitled%204.png)

![Untitled](Examples%207c0ce7a52f834d068963e03e650ac3a3/Untitled%205.png)

![Untitled](Examples%207c0ce7a52f834d068963e03e650ac3a3/Untitled%206.png)

![Untitled](Examples%207c0ce7a52f834d068963e03e650ac3a3/Untitled%207.png)

# Finding Parzen PDF

![Untitled](Examples%207c0ce7a52f834d068963e03e650ac3a3/Untitled%208.png)

# Estimating Covariance Matrix

![Untitled](Examples%207c0ce7a52f834d068963e03e650ac3a3/Untitled%209.png)