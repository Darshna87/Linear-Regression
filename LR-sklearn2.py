import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression

dataset=load_breast_cancer()
X=dataset.data
Y=dataset.target
print(X.shape,Y.shape)

print(dataset.feature_names)
#print(dataset.DESCR)
print(X[:5,:5])
print(Y[:5])

model=LinearRegression()
model.fit(X,Y)

print("Coeffeicients : ",model.coef_)
print("Intercept : ",model.intercept_)

print("Predictions by model (for first 5 values):")
print(model.predict([X[1],X[2],X[3],X[4],X[5]]))
print("Actual predictions: ")
print(Y[:5])

print("Score : ",model.score(X,Y))



