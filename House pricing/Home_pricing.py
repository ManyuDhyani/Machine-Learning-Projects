#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn import linear_model


# In[29]:


#price = m1*area + m2*bedrooms + m3*age + b
#price = m1x1 + m2x2 + m3x3 + b


# In[30]:


df = pd.read_csv('data.csv')
df


# In[31]:


df.bedrooms.median()


# In[32]:


df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())
df


# In[33]:


reg = linear_model.LinearRegression()
reg.fit(df.drop('price', axis='columns'), df.price)


# In[34]:


reg.coef_


# In[35]:


reg.intercept_


# In[36]:


reg.predict([[3000, 3, 40]])


# In[37]:


reg.predict([[2500, 4, 5]])

