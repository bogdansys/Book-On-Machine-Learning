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