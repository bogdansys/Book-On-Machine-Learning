# Python Usage

## [Estimating class conditional probabilities](https://weblab.tudelft.nl/cse2510/2022-2023/assignment/99902/)

```java
Given that a M
-dimensional Gaussian pdf is given by:
p(x)=1(2π)Mdet(Σ)−−−−−−−−−−√exp(−12(x−μ)TΣ−1(x−μ))
and given a 1-dimensional trainingset with n=4
 datapoints:
x1=1.2
, x2=3.4
, x3=2.7
, x4=4.5
.
Implement this probability density function p(x)
 in Python where the parameters μ
 and Σ
 are estimated from the given training set. The function should output the value of p(x)
.

If you want, you may use numpy-functions like numpy.mean, numpy.std, numpy.var, numpy.linarg.det, numpy.inv, numpy.sqrt, numpy.exp, numpy.log, numpy.power, etc.etc.
```

```python
import numpy as np

class Solution():

  def solution(testx):
    traindata = [ 1.2, 3.4, 2.7, 4.5]

    mu = np.mean(traindata)
    var = np.var(traindata)

    px = np.exp(-0.5*(testx-mu)**2/var)/np.sqrt(2.*np.pi*var)

    return px
```

# K nn majority vote

```python
Given an array of nearest neighbors and the list of labels for each instance in the array of the nearest neighbours, implement a function that performs majority vote. Return the label of the most common class. If it is a tie (i.e. multiple classes are equally common), your function should only return the first class encountered.

Input:
- x: numpy array with the indices of nearest neighbors
- y: numpy array of labels

Output:
- label of the most common class
```

```python
import numpy as np

class Solution():
  
 def nearest_neighbors(x, y):
    freq = {}
    for i in x:
        if y[i] in freq:
            freq[y[i]] += 1
        else:
            freq[y[i]] = 1
    return max(freq, key=freq.get)

```

# These are past exam questions and they recycle them  (there are like this type only)