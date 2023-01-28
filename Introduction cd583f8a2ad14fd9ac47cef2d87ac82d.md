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