# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:


Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: CHANDRAPRIYADHARSHINI C
RegisterNumber:  212223340019
```

import pandas as pd
data=pd.read_csv("/content/Employee (1).csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:


![ml 1](https://github.com/Bosevennila/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870486/541c24b3-59d2-48d7-81f3-d120d285dd2f)
![ml 2](https://github.com/Bosevennila/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870486/5dd19e9d-6f46-44b1-aad0-fb75131a25fa)
![ml 3](https://github.com/Bosevennila/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870486/2272c806-838b-4d04-aeb1-82fc809045d4)
![ml 4](https://github.com/Bosevennila/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870486/c0437120-e318-4285-b86d-7a561abdd8bf)
![ml 5](https://github.com/Bosevennila/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870486/e0105ab9-f922-4de5-97ae-c7f296cd8ec0)
![ml 6](https://github.com/Bosevennila/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870486/5769c780-813f-4d20-8c9a-e646b4b6a90e)
![ml 7](https://github.com/Bosevennila/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870486/52d5e1c2-a100-4458-a3a6-f45e62306974)
![ml 8](https://github.com/Bosevennila/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/144870486/df2b9024-c189-4566-8a9a-a953d5229f2b)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
