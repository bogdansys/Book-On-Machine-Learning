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