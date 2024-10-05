#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


dataset = pd.read_csv(r"D:\Social_Network_Ads1.csv")
dataset.drop(columns = ["EstimatedSalary"],inplace = True)


# In[8]:


dataset.head(3)


# In[12]:


plt.figure(figsize=(4,3))
sns.scatterplot(x = "Age", y = "Purchased", data = dataset)
plt.show()


# In[13]:


x = dataset [["Age"]]
y = dataset ["Purchased"]


# In[14]:


from sklearn.model_selection import train_test_split


# In[51]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)


# In[52]:


from sklearn.linear_model import LogisticRegression


# In[53]:


lr = LogisticRegression()
lr.fit(x_train,y_train)


# In[54]:


lr.score(x_test,y_test)*100


# # Predict

# In[57]:


lr.predict([[40]])


# In[62]:


plt.figure(figsize=(6,5))
sns.scatterplot(x = "Age", y = "Purchased", data = dataset)
sns.lineplot(x = "Age", y = lr.predict(x), data = dataset, color = "red")
plt.show()


# In[ ]:




