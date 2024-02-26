# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and read the dataframe.

2.Assign hours to X and scores to Y.

3.Implement training set and test set of the dataframe

4.Plot the required graph both for test data and training data.

5.Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JAYAVARTHAN P
RegisterNumber: 212222100015

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
## Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
## Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

*/
```

## Output:
![ML2](https://github.com/JAYAVARTHAN-P/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121369281/e78ebfc4-e8a6-4017-ac46-42e1496dfc58)


![ml 3](https://github.com/JAYAVARTHAN-P/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121369281/8296af53-60dc-43bb-9118-7b7a34e13686)


![ML4 0](https://github.com/JAYAVARTHAN-P/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121369281/2fe5345a-59d6-4ece-96f6-3984e16de178)


![ML4](https://github.com/JAYAVARTHAN-P/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121369281/4eafef3a-b3d4-448d-8bb4-ebff861b2aab)


![FIN](https://github.com/JAYAVARTHAN-P/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121369281/84f5cb57-c3f2-48d6-8733-223e0609c927)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
