import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


data= pd.read_csv("D:\ML challenges\linear regression\Real estate.csv")

data

data.head(2)

data.describe()

data.info()

data.isnull().sum()

data.dropna()

data.mean()

data.median()

import matplotlib.pyplot as plt

plt.boxplot(data)
plt.show()

numeric_column=data.select_dtypes(include='number').columns.tolist()

import math
num_plots=len(numeric_column)
num_cols=3
num_rows=math.ceil(num_plots/num_cols)

plt.figure(figsize=(15,20))
for i,col in enumerate(numeric_column,1):
    plt.subplot(num_rows,num_cols,i)
    data.boxplot(column=col)
    plt.title("box plot")

import seaborn as sns
corr=data.select_dtypes(include='number').corr()
sns.heatmap(corr,annot=True)
plt.show()

sort=np.sort(data['X5 latitude'] )
Q1 = np.percentile(data['X5 latitude'], 25)
Q1

sort=np.sort(data['X5 latitude'] )
Q2 = np.percentile(data['X5 latitude'], 50)
Q2

col=data.select_dtypes(include='number').columns.tolist()
features=data[col]
features.drop('Y house price of unit area',axis=1,inplace=True)
target=data['Y house price of unit area']

features

target

import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,target,train_size=0.8,random_state=42)

import sklearn.linear_model as lm
from sklearn.linear_model import LinearRegression
model= LinearRegression()
model.fit(x_train,y_train)

prediction=model.predict(x_test)

import sklearn.metrics as mt
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,prediction)
print(mse)

import joblib
joblib.dump(model, "linear_regression_model.pkl")

