import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

##  Load training set
iris = load_iris()
print(iris.data)
print(iris.target)
iris_X, iris_y = iris.data[:-1,:], iris.target[:-1] ## iris.data is 2 dim so slicing need 2 argument (-1 mean size-1)
iris_y= pd.get_dummies(iris_y).values               ## create vector of output(2 -> [0 1 0])
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)
                                                    ## split to train and test
print(iris_y)

# numFeatures is the number of features in our input data.
# In the iris dataset, this number is '4'.
numFeatures = trainX.shape[1]                       ## shape[0] -> num of row(height)
                                                    ## shape[1] -> num of col(width)
# numLabels is the number of classes our data points can be in.
# In the iris dataset, this number is '3'.
numLabels = trainY.shape[1]


# Placeholders
# 'None' means TensorFlow shouldn't expect a fixed number in that dimension
X = tf.placeholder(tf.float32, [None, numFeatures]) # Iris has 4 features, so X is a tensor to hold our data.
yGold = tf.placeholder(tf.float32, [None, numLabels]) # This will be our correct answers matrix for 3 classes.

