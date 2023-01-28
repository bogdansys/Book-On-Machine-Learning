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