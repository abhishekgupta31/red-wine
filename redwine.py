#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('wine_dataset.csv')


# In[3]:


df.head(10)


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.describe


# In[7]:


df.isnull().sum()


# In[8]:


df.skew()


# In[9]:


import sklearn
import seaborn as sns
import matplotlib.pyplot as plt


# In[12]:


col = df.columns.values
ncol = 30
nrow=14
plt.figure(figsize=(ncol,5*nrow))
for i in range(0,len(col)):
    plt.subplot(nrow,ncol,i+1)
    sns.boxplot(df[col[i]],color='red',orient='v')
    plt.tight_layout()


# In[13]:


sns.distplot(df['fixed acidity'])


# In[14]:


sns.distplot(df['volatile acidity'])


# In[15]:


sns.distplot(df['citric acid'])


# In[16]:


sns.distplot(df['residual sugar'])


# In[17]:


sns.distplot(df['chlorides'])


# In[18]:


sns.distplot(df['free sulfur dioxide'])


# In[19]:


sns.distplot(df['total sulfur dioxide'])


# In[20]:


sns.distplot(df['density'])


# In[21]:


sns.distplot(df['sulphates'])


# In[22]:


sns.distplot(df['alcohol'])


# In[23]:


plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(),annot=True)


# In[24]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from sklearn.metrics import accuracy_score


# In[41]:


x = df.drop(('quality'),axis=1)
y = df['quality']


# In[26]:


print(x.shape)
print(y.shape)


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[42]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[32]:


lr = LogisticRegression()


# In[33]:


lr.fit(x_train,y_train)


# In[43]:


x_train.columns


# In[44]:


print(lr.coef_)


# In[46]:


y_pred = lr.predict(x_test)
print(y_pred)


# In[47]:


print("F1 Score :",f1_score(y_pred,y_test,average = "weighted"))
print('Report:\n',classification_report(y_test, y_pred))


# In[48]:


confusion_matrix(y_test, y_pred)


# In[51]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[53]:


corrs = x_test.corr()


# In[54]:


plt.figure(figsize=(10,10))
sns.heatmap(corrs,cmap='Reds',annot=True)
plt.show()


# In[ ]:




