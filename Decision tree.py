#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split  # fixed 'tarin_test_split' typo
from sklearn.tree import DecisionTreeClassifier  # fixed 'sklaern' typo
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[8]:


iris = datasets.load_iris(as_frame=True).frame


# In[9]:


iris = pd.read_csv("iris.csv")
print(iris)


# In[10]:


labelencoder = LabelEncoder()
iris.iloc[:, -1] = labelencoder.fit_transform(iris.iloc[:, -1])
iris.head()


# In[5]:


iris.info()


# In[11]:


iris['variety'] = pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[13]:


iris.head(3)


# In[6]:


iris[iris.duplicated(keep= False)]


# In[22]:


X=iris.iloc[:,0:4]
Y=iris['variety']


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 1)
x_train


# Building Decision Tree Classifier using Entropy Criteria

# In[24]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth =None)
model.fit(x_train,y_train)


# In[32]:


plt.figure(dpi=1200)
tree.plot_tree(model);


# In[ ]:




