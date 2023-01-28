# Empirical Risk Minimisation

![Untitled](Empirical%20Risk%20Minimisation%201ad47dae35e54609b98feec8502bc6dd/Untitled.png)

Empirical risk minimization (ERM) is a principle used in statistical learning and machine learning to minimize the generalization error of a model on unseen data. It is used to find the best parameters of a model that minimize the difference between the predicted values and the true values on the training data.

The basic idea of ERM is to define a loss function, which measures the difference between the predicted and true values, and then find the set of parameters that minimize the average value of this loss function over the training data. This can be done using optimization algorithms such as gradient descent.

ERM is widely used in supervised learning, where the goal is to find the best model that can accurately predict the target variable based on the input features. The choice of the loss function depends on the type of problem, for example, for classification problems, the cross-entropy loss function is commonly used, while for regression problems the mean squared error loss function is used.

ERM has several advantages, including its simplicity and ease of implementation. It can be used with a wide range of different models and it is easy to adapt to different types of data and different types of problems. However, it is worth noting that ERM can be sensitive to the choice of the loss function, and it can be affected by the presence of noise or outliers in the data.

In summary, Empirical Risk Minimization is a principle used in statistical learning and machine learning to minimize the generalization error of a model on unseen data, it uses a loss function to measure the difference between the predicted and true values and it finds the set of parameters that minimize the average value of this loss function over the training data. It is widely used in supervised learning and it has several advantages, but it is sensitive to the choice of the loss function and it can be affected by the presence of noise or outliers in the data.