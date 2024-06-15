#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


df=pd.read_csv('salary.csv')


# In[26]:


df.head()


# In[27]:


plt.scatter(df['YearsExperience'],df['Salary'])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")


# In[28]:


X = df[['YearsExperience']]
Y = df['Salary']


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.25, random_state=42)


# In[30]:


## Standardization
from sklearn.preprocessing import StandardScaler


# In[36]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[37]:


from sklearn.linear_model import LinearRegression


# In[40]:


regression = LinearRegression(n_jobs = -1)
regression.fit(X_train,Y_train)
print("Coefficient: ",regression.coef_)
print("Intercept: ",regression.intercept_)


# In[43]:


plt.scatter(X_train,Y_train)
plt.plot(X_train,regression.predict(X_train))


# In[44]:


Y_pred = regression.predict(X_test)


# In[45]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[47]:


mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test,Y_pred)
rmse = np.sqrt(mse)
print(mse)
print(mae)
print(rmse)


# In[48]:


from sklearn.metrics import r2_score
score = r2_score(Y_test, Y_pred)
print(score)


# In[ ]:




