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