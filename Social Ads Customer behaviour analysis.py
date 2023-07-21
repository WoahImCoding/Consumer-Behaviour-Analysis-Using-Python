#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().system('pip install sklearn')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScal


# In[10]:


dataset = pd.read_csv('Social_Network_Ads.csv')
print(dataset)
dataset.shape
X = dataset.iloc[:, :-1].values  #which simply means take all rows and all columns except last one
y = dataset.iloc[:, -1].values   #which simply means take all rows and only columns with last column


# In[11]:


# Exploratory Data Analysis


# In[12]:


dataset.info()


# In[13]:


# Splitting the dataset into Training Set and Test Set
# For this we will use the train_test_split method from library model_selection 
# We are providing a test_size of 0.25 (for better performance because here dataset contain large amount of data)
# which means test set will contain 100 observations and training set will contain 300 observations.


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[15]:


print(X_train)  # X_train dataset containg 300 observation for training 


# In[16]:


print(y_test)  # y_test dataset containg 100 observation for testing 


# In[17]:


# Using Feature Scaling for better prediction, equalising datas at same range


# In[54]:


# Separate features and target
X = dataset.drop(columns=['Purchased']).values
y = dataset['Purchased'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_test)
#By using pd.get_dummies() with drop_first=True, we ensure that the 'Gender' column is properly encoded as numerical values. Then, the StandardScaler can handle the data without any issues. The code should work as expected, and you should see the scaled X_test dataset printed without any errors.







# In[ ]:


#Customer Behaviour Analysis Using Logistic Regression


# In[44]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[46]:


# Predict Result
print(classifier.predict(sc.transform([[30,87000,0,0]])))


# In[ ]:


# The test result is 1 which means the customer will Buy


# In[ ]:


# Predicting the test set results


# In[47]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[48]:


# Making Confusion Matrix, To test our Models precision score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:


#Taking Input value from the user
print("ENTER CUSTOMER AGE AND EstimatedSalary TO FIND OUT CUSTOMER WILL BUY CAR OR NOT:")

age=int(input("Age: "))
salary=int(input("EstimatedSalary: "))
result = classifier.predict(sc.transform([[age,salary]]))
print(result)
if result==[1]:
    print("Customer Will Buy Car")
else:
    print("Customer Will Not Buy Car")


# In[ ]:




