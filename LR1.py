import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url='https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'
df=pd.read_csv(url)
print(df.head())

#load values

df_array=df.values
#print(df_array)

X=df_array[:,0]
Y=df_array[:,1]
print(X.shape)
print(Y.shape)

#normalise
u=X.mean()
std=X.std()

print("Mean:",u)
print("Std:",std )
X=(X-u)/std
print(X)

#visualize
plt.style.use("dark_background")

plt.scatter(X,Y)
plt.title("Student performance")
plt.xlabel("No of Hours")
plt.ylabel("performance")
plt.show()


#Step2: Training Stage

def hypothesis(x,theta):

  y_=theta[0]+theta[1]*x
  return y_

def Gradient(X,Y,theta):

  m=X.shape[0]
  grad=np.zeros((2,))

  for i in range(m):

    y_ = hypothesis(X[i],theta)
    grad[0] += y_-Y[i]
    grad[1] +=(y_-Y[i])*X[i]

  return grad/m


def Error(X, Y, theta):
    m = X.shape[0]
    total_error = 0

    for i in range(m):
        y_ = hypothesis(X[i], theta)
        total_error += (y_ - Y[i]) ** 2

    J_theta = total_error / m
    return J_theta


def GradientDecent(X, Y, steps=100, learning_rate=0.1):
    theta = np.zeros((2,))
    errorlist = []
    thetalist = []

    for i in range(steps):
        grad = Gradient(X, Y, theta)
        J_theta = Error(X, Y, theta)

        theta[0] = theta[0] - learning_rate * grad[0]
        theta[1] = theta[1] - learning_rate * grad[1]

        thetalist.append((theta[0], theta[1]))
        errorlist.append(J_theta)

    return theta, errorlist, thetalist


theta,errorlist,thetalist=GradientDecent(X,Y)
print(theta)
print(errorlist)
print(thetalist)

plt.plot(errorlist)
plt.show()

# step3: Visualizing Best fit line

y_=hypothesis(X,theta)
print(y_)

plt.scatter(X,Y)
plt.plot(X,y_,color="red")
plt.show()

#step4 :Score computation
#R squared/R2 score  :Applied on Training data

def score(y,y_):
  num=np.sum((y-y_)**2)
  denom=np.sum((y-y.mean())**2)
  sr=1-num/denom
  return sr

result=score(Y,y_)*100
print(result,"%")


# step5 :visualization of Loss Function ,Gradient Decent and theta updates

 #Loss Function Visualization

 print(theta)
#preparing plane by taking 100 points based on theta0 and theta1 values

T0=np.arange(25,75,1)
T1=np.arange(0,50,1)
#print(T0,T1)

T0,T1=np.meshgrid(T0,T1)
J=np.zeros(T0.shape)

for i in range(T0.shape[0]):
  for j in range(J.shape[0]):
        Y_=T1[i,j]*X+T0[i,j]
        J[i,j]=np.sum((Y-Y_)**2)/Y.shape[0]

#J=T0+T1
fig = plt.figure()
axes = fig.gca(projection='3d')
axes.plot_surface(T0,T1,J,cmap="rainbow")
plt.show()

fig = plt.figure()
axes = fig.gca(projection='3d')
axes.contour(T0,T1,J,cmap="rainbow")
plt.show()


# theta updates visualization

theta_list=np.array(thetalist)
#print(theta_list)
plt.figure()
plt.plot(theta_list[:,0],label='Theta0',color="red")
plt.plot(theta_list[:,1],label='Theta1',color='yellow')
plt.legend()
plt.show()

#Trajectory of Loss function with theta values i.e. Gradient Decent
error_list=np.array(errorlist)
fig = plt.figure(figsize=(10,10))
axes = fig.gca(projection='3d')
axes.plot_surface(T0,T1,J,cmap="rainbow")
plt.scatter(theta_list[:,0],theta_list[:,1],error_list)
plt.show()

fig = plt.figure(figsize=(10,10))
axes = fig.gca(projection='3d')
axes.contour(T0,T1,J,cmap="rainbow")
plt.scatter(theta_list[:,0],theta_list[:,1],error_list)
plt.show()

#for interactive plot
np.save('theta.npy',theta_list)
theta1=np.load('theta.npy')
T0=theta1[:,0]
T1=theta1[:,1]

plt.ion()
for i in range(0,50,3):
  y_=T1[i]*X+T0[i]
  plt.scatter(X,Y)
  plt.plot(X,y_,'red')
  plt.draw()
  plt.pause(1)
  plt.clf()