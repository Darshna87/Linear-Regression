"""
#locally weighted Regression

1. Load and normalize data
2. Generate W for every query point
3. No training is needed, directly make predictions using "Closed Form Solution"
  theta=(X'WX).inverse.X'WY where X'=X_transpose
4. find the best value of Tau(bandwidth parameter) (cross validation)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load data

dfx=pd.read_csv("..//content/sample_data/WeightedX.csv")
dfy=pd.read_csv("..//content/sample_data/WeightedY.csv")

X=dfx.values
Y=dfy.values
print(X.shape,Y.shape)

# normalize data

u=X.mean()
std=X.std()
X=(X-u)/std

M=X.shape[0]
print(M)

plt.style.use("seaborn")
plt.scatter(X,Y)
plt.show()

#2. Generate weight Matrix

def GetW(query_point,X,tau):
  #m=X.shape[0]
  W=np.eye(M)
  
  for i in range(M):
    xi=X[i]
    x=query_point
    W[i,i]=np.exp(np.dot((xi-x),(xi-x).T)/(-2*tau*tau))
  return W

X=np.mat(X)
Y=np.mat(Y)
W=GetW(-1,X,0.05)
print(W)

"""# Step 3: Predictions"""

def predict(X,Y,query_point,tau):


      ones=np.ones((M,1))
      X_=np.hstack((X,ones))

      qry_pt=np.mat([query_point,1])


      W=GetW(qry_pt,X_,tau)
    
      #theta=(X'WX).inverse.X'WY 
      #         term1       term2
      
      term1=np.dot(X_.T,np.dot(W,X_))
      term2=np.dot((X_.T),np.dot(W,Y))
      
      theta=np.dot(np.linalg.pinv(term1),term2)

      pred=np.dot(qry_pt,theta)

      return theta,pred

theta,pred=predict(X,Y,1.5,1.0)
print(theta,pred)

"""# Visualize Predicitons 
#analyse effects of Tau
"""

def plotPrediction(tau):

  X_test=np.linspace(-2,2,20)
  Y_test=[]

  for qx in X_test:
    theta,pred=predict(X,Y,qx,tau)
    Y_test.append(pred)
  
  Y_test=np.array(Y_test)

  XO=np.array(X)
  YO=np.array(Y)

#  plt.style.use("dark-background")
  plt.title("Tau /bandwidth value %.2f"%tau)
  plt.scatter(XO,YO)
  plt.scatter(X_test,Y_test,color="red")
  plt.show()

  return

#Analyse the effect of Tau values

tau_values=[0.1,0.5,1,2,5,10]

for tau in tau_values:
  plotPrediction(tau)

