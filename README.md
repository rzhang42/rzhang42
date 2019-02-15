# rzhang42
Digit Recognizer

Use the famous MNIST data

Description

MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

Skills

Computer vision fundamentals including simple neural networks
Classification methods such as SVM and K-nearest neighbors

Goal

The goal is to take an image of a handwritten single digit, and determine what that digit is.
For every ImageId in the test set, I need to predict the correct label. 

Introduction

The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.
Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.
The training data set has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image. Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783.
The test data set is the same as the training set, except that it does not contain the "label" column.
More details about the dataset, can be found at http://yann.lecun.com/exdb/mnist/index.html. The dataset is made available under a Creative Commons Attribution-Share Alike 3.0 license.
