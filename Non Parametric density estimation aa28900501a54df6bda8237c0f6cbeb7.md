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