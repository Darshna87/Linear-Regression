
# Boston Houseing Dataset

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading boston dataset

boston=load_boston()
X=boston.data
Y=boston.target
print(X.shape,Y.shape)

print(boston.feature_names)
print(boston.DESCR)

df=pd.DataFrame(X)
df.columns=boston.feature_names
print(df.head())

df.describe()

#from description mean is not zero.thus data needs to be normalized
# finding mean and standard deviation along with columns

u=np.mean(X,axis=0)
std=np.std(X,axis=0)

print(u.shape,std.shape)
X=(X-u)/std
print(pd.DataFrame(X[:5,:5]))

#ploting Y vs 5th feauter
plt.style.use('seaborn')
plt.scatter(X[:,5],Y)
plt.show()


#Data preparation
# adding 0th columns of ones for theta0 multiplication

t0=np.ones((X.shape[0],1))
print(t0.shape)

X=np.hstack((t0,X))
print(X[:4,:4])

# Notation:      X:m*n matrix,   x: single Example,    y:vector

def hypothesis(x,theta):

  y_=0.0
  n=x.shape[0]
  for i in range(n):
    y_+=theta[i]*x[i]

  return y_


def Error(X, y, theta):
    total_error = 0.0
    m = X.shape[0]

    for i in range(m):
        y_ = hypothesis(X[i], theta)
        total_error += (y[i] - y_) ** 2

    J_theta = total_error / m
    return J_theta


def Gradient(X, y, theta):
    m, n = X.shape
    # m=X.shape[0]
    # n=X.shape[1]
    grad = np.zeros((n,))

    # calculating for n gradients
    for j in range(n):

        # calculating value of Jth gradient
        for i in range(m):
            y_ = hypothesis(X[i], theta)
            grad[j] += (y_ - y[i]) * X[i][j]

    return grad / m

def GradientDecent(X,y,learning_rate=0.1,Epochs=300):

    m,n=X.shape
    theta=np.zeros((n,))
    error_list=[]
    grad=np.zeros((n,1))

    for i in range(Epochs):

      error=Error(X,y,theta)
      error_list.append(error)

      grad=Gradient(X,y,theta)
      for j in range(n):
        theta[j]=theta[j]-learning_rate*grad[j]

    return theta,error_list

import time
start=time.time()

theta,error_list=GradientDecent(X,Y)
end=time.time()

print(end-start)
print(theta)
#quite slow --20 seconds

#print(error_list)
plt.plot(error_list)
#R2 squared on predictions/ score calculation
def score(y,y_):
  num=np.sum((y-y_)**2)
  denom=np.sum((y-y.mean())**2)
  sr=(1-num/denom)
  return sr*100

y_=[]
#m=Y.shape[0]
#print(m)
for i in range(m):
  pred=hypothesis(X[i],theta)
  y_.append(pred)

y_=np.array(y_)
print(score(Y,y_))



# Section 3: Optimization
###-Avoid loops Except main loop of Gradient Decent
###-Use np.sum() and np.dot() which are quite fast and already optimized

def hypothesis1(X,theta):
  return np.dot(X,theta)

def error1(X,y,theta):

  e=0.0
  m=X.shape[0]

  y_=hypothesis1(X,theta)
  e=np.sum((y-y_)**2)

  return e/m


def gradient1(X,y,theta):

  y_=hypothesis1(X,theta)
  grad=np.dot(X.T,(y_-y))
  m=X.shape[0]

  return grad/m


def gradientDecent1(X, y, learning_rate=0.1, epochs=300):
    errorlist = []
    n = X.shape[1]
    theta = np.zeros((n, 1))

    for i in range(epochs):
        e = error1(X, y, theta)
        errorlist.append(e)

        grad = gradient1(X, y, theta)
        theta = theta - learning_rate * grad

    return theta, errorlist

start=time.time()
theta,errorlist=gradientDecent1(X,Y)
end=time.time()
print(end-start)

print(theta)
plt.plot(errorlist)

#predicitons
y_=hypothesis1(X,theta)
print(score(Y,y_))