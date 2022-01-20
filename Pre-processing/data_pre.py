#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


dataset = pd.read_csv('D:\Data.csv')


# In[4]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[5]:


print(X)


# In[6]:


print(y)


# In[7]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# In[8]:


print(X)


# In[9]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[10]:


print(X)


# In[11]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[12]:


print(y)


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[14]:


print(X_train)


# In[15]:


print(X_test)


# In[16]:


print(y_train)


# In[17]:


print(y_test)


# In[18]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.fit_transform(X_test[:, 3:])


# In[19]:


print(X_train)


# In[21]:


print(X_test)


# In[ ]:




