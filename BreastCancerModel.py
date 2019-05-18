import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
dataset= datasets.load_breast_cancer()

x=dataset.data[:,:]
y=dataset.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

from sklearn.preprocessing import StandardScaler
std=StandardScaler()
X_train=std.fit_transform(x_train)
X_test=std.transform(x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

clf=LogisticRegression()
#clf=LinearRegression()

x_train=x_train.reshape(-1,1)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_test,y_pred)

m=clf.coef_
c=clf.intercept_
clf.score(X_test,y_test)