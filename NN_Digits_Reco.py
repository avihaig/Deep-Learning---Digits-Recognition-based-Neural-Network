#Avi Giuili - ID : 330190950

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import  matplotlib.pyplot as plt

# 1) IMPORT THE MNIST database
mnist = fetch_openml('mnist_784',version=1,return_X_y=False,as_frame=False)
X = mnist['data'].astype('float64')
target = mnist['target']
random_state = check_random_state(1)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
target = target[permutation]

# 2) Flatten the database example images to 1D vector of length 784 
X = X.reshape((X.shape[0], -1))

# 4) Divide the database to 3 parts: training set (60%), validation set (20%) and test set (20%):
X_train, X_testandval, y_train, y_testandval = train_test_split(X, target, test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_testandval, y_testandval, test_size=0.5)

# the next lines standardize the images 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_val = scaler.fit_transform(X_val)
X_test = scaler.transform(X_test)

# 3) add the weights bias 1 at the end of each 1-dimensional vector of length 784 (to length of 785)
X_train = np.hstack((X_train,np.ones((X_train.shape[0],1))))
X_val = np.hstack((X_val,np.ones((X_val.shape[0],1))))
X_test = np.hstack((X_test,np.ones((X_test.shape[0],1))))


# 6) Write the error function in a matrix form : cross entropy loss

#'one-hot' format for every single tag
y_train = np.eye(10)[y_train.astype(int)]
y_val = np.eye(10)[y_val.astype(int)]
y_test = np.eye(10)[y_test.astype(int)]


# 5) Initialize random weights vectors W1, W2 ,...,W9 ,of length 785
w = np.random.rand(10,785)

#predicted tag of the n-th example
def Y_n_k(W,X):
    y_n_k = np.zeros((X.shape[0], 10))
    for n in range(y_n_k.shape[0]):
        for k in range(y_n_k.shape[1]):
            y_n_k[n, k]= np.exp(np.dot(W[k], X[n]))
            denominator= 0
            for j in range(10):
                denominator += np.exp(np.dot(W[j], X[n]))
            y_n_k[n, k] = y_n_k[n,k] / denominator
    return y_n_k

def error_function(y_n_k,t_n_k):
    return -np.sum(t_n_k*np.log(y_n_k))

# 7) Minimize the error function using the gradient descent algorithm

def gradient(x_n,y_n_j,t_n_j):
    grad_E = np.zeros((10,785))
    for j in range(10):
        grad_E[j] = np.matmul(np.transpose(x_n) ,y_n_j[:,j] - t_n_j[:,j])
    return grad_E

def model_accuracy(W,X,t):
    return (np.sum(np.equal(np.argmax(Y_n_k(W,X),1),np.argmax(t,1)))/X.shape[0])*100

η = 0.0001
training_loss= np.zeros((10,1))
accuracy = np.zeros((10,1))
for j in range(10):
    y = Y_n_k(w, X_train)
    grad = gradient(X_train,y,y_train)
    # b) for each iteration, calculate the loss on the training set
    training_loss[j] = error_function(y,y_train)
    # c) for each iteration, calculate the model accuracy on validation test
    accuracy[j] = model_accuracy(w, X_val, y_val)
    # e) stop running when the accuracy on the validation set does not change much anymore
    if ((j > 0) and (accuracy[j] - accuracy[j-1] < 1)):
        break
    # a) for each iteration, update : Wj(r+1) = Wj(r) - η*grad(E(Wj(r)))
    w = w - η*grad

# 10) write the accuracy obtain on each part of the database
print("accuracy obtained on the training set : ", model_accuracy(w,X_train,y_train))
print("accuracy obtained on the Validation set : ", accuracy[j])
print("accuracy obtained on the test set : ", model_accuracy(w,X_test,y_test))

# 8) plot Loss function on the training set
plt.figure(1)
plt.plot(training_loss[0:j+1])
plt.title("Loss function on training set")

# 9) plot accuracy obtain on the validation set
plt.figure(2)
plt.plot(accuracy[0:j+1])
plt.title("Accuracy on validation set")
plt.show()