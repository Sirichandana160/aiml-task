#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np


# In[10]:


df = pd.read_csv("Universities.csv")
df


# In[11]:


# M<ean value of SAT score
np.mean(df["SAT"])


# In[12]:


# Median of the data
np.median (df["SAT"])


# In[13]:


# Standard deviation of data
np.std(df["GradRate"])


# In[14]:


# Find the variance
np.var(df["SFRatio"])


# In[15]:


df.describe


# # visualizations

# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


plt.figure(figsize=(6,3))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# # Observations

# # visualization using boxplot

# In[27]:


s = [20,15,10,25,30,35,28,40,45,60]
scores = pd.Series(s)
scores


# In[28]:


plt.boxplot(scores, vert=False)


# In[29]:


s = [20,15,10,25,30,35,28,40,45,60,120,150]
scores = pd.Series(s)
scores


# In[30]:


plt.boxplot(scores, vert = False)


# # Identify outliers in universities dataset

# In[31]:


df = pd.read_csv("universities.csv")
df


# In[32]:


plt.boxplot(df["SAT"])


# In[34]:


plt.figure(figsize=(6,2))
plt.title("Box plot for SAT Score")
plt.boxplot(df["SAT"], vert = False)


# In[ ]:





# In[ ]:





# In[ ]:




