import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
#for 3d projection
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression

X,Y=make_regression(n_samples=1000,n_features=2,n_informative=1,noise=10,random_state=1)
print(X.shape,Y.shape)

plt.subplot(1,2,1)
plt.scatter(X[:,0],Y)
plt.subplot(1,2,2)
plt.scatter(X[:,1],Y)
#plt.show()

#3d projection with X0,X1,and Y features

fig = plt.figure(figsize=(10,10))
axs = plt.axes(projection='3d')
axs.scatter3D(X[:,0],X[:,1],Y,color="red")
plt.title("3D scatter plot")
plt.show()

model=LinearRegression()
model.fit(X,Y) # train the model

#after training values are stored in model object which can be  accessed through model

print(model.coef_) #theta1 and theta2
print(model.intercept_) #theta0

print("Y_[0],Y_[1] : ",model.predict([X[0],X[1]]))
print("Y[0],Y[1] : ",(Y[0],Y[1]))
print("Score:",model.score(X,Y))