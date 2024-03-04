# imported to bring the data in, copy-patse from code given in excersize
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
#numpy is used to speed up calculations
import numpy as np
# matplotlib is used to draw graphs
import  matplotlib.pyplot as plt
# sub question 6 - loss function calculation
def loss(y,t):
#   certain numpy function contain internal elementwise loops
#   so some things are nut necessary
    E = -np.sum(t*np.log(y))
    return E
# general function to calculate y
def calcY(W,X):
    # start by creating array
    y = np.ones((X.shape[0], 10))
    #   start calculating for each n and k
    for n in range(0, X.shape[0]):
        for k in range(0, 10):
            d = np.dot(W[k], X[n])
            y[n, k] = np.exp(d)
            denom = 0
            for j in range(0, 10):
                denom += np.exp(np.dot(W[j], X[n]))
            y[n, k] = y[n, k] / denom
    return y

# gets the model's accuracy
def Accuracy(W,X,t):
    y = calcY(W,X)
    guess = np.argmax(y,1)
    actual = np.argmax(t,1)
    countRight = np.sum(np.equal(guess,actual))
    return countRight/X.shape[0]*100

# get the gradient
def gradient(x,y,t):
    # memory
    grad = np.ones((10,785))
    for j in range(0,10):
        grad[j] = np.matmul(np.transpose(x),y[:,j] - t[:,j])
    return grad

# the "main" code
# Sub Question 1: MNIST importation from class code, the first line was changed to prevent errors
mnist = fetch_openml('mnist_784',version=1,return_X_y=False,as_frame=False)
X = mnist['data'].astype('float64')
t = mnist['target']
random_state = check_random_state(1)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
t = t[permutation]

# Sub Questions 2 and 3: flatten and create matrix
X = X.reshape((X.shape[0], -1)) #This line flattens the image into a vector of size 784
# Sub Question 4: split, first to train and test
X_train, X_rest, t_train, t_rest = train_test_split(X, t, test_size=0.4)
# now break the test in two for validation
X_val, X_test, t_val, t_test = train_test_split(X_rest, t_rest, test_size=0.5)
# The next lines standardize the images
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.transform(X_test)
# add the ones at the end
X_train = np.hstack((X_train,np.ones((X_train.shape[0],1))))
X_val = np.hstack((X_val,np.ones((X_val.shape[0],1))))
X_test = np.hstack((X_test,np.ones((X_test.shape[0],1))))
# turn t into one hot vectors
t_train = np.eye(10)[t_train.astype(int)]
# turn t into one hot vectors
t_val = np.eye(10)[t_val.astype(int)]
# turn t into one hot vectors
t_test = np.eye(10)[t_test.astype(int)]
# sub question 5: initialize the weights matrix
w = np.random.rand(10,785)
# initialize eta
eta = 0.00004
# sub question 7: gradient descent
trainLoss = np.ones((10,1))
accuracy = np.ones((10,1))
for j in range(0,10):
    # training y should be calculated once to save time
    y = calcY(w, X_train)
    trainLoss[j] = loss(y,t_train)
    accuracy[j] = Accuracy(w, X_val, t_val)
    grad = gradient(X_train,y,t_train)
    # a stop condition as requested
    if ((j > 0) and (accuracy[j] - accuracy[j-1] < 1)):
        break
    w = w - eta*grad
print("Validation:")
print(accuracy[j])
print("training:")
print(Accuracy(w,X_train,t_train))
print("testing:")
print(Accuracy(w,X_test,t_test))
plt.figure(1)
plt.plot(trainLoss[0:j+1])
plt.title("the loss function of the training set")
plt.figure(2)
plt.plot(accuracy[0:j+1])
plt.title("the accuracy of the validation set")
plt.show()