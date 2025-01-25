#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[15]:


# Drop duplicated rows
data.drop_duplicates(keep='first', inplace = True)
data


# In[16]:


# Change column names (Rename the columns)
data.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data


# In[17]:


# Display data1 missinmg values count in each column using isnull().sum()
data.isnull().sum()


# In[18]:


# visualize data1 missing values using heat map

cols = data.columns
colors = ['black', 'yellow']
sns.heatmap(data[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[19]:


# Find the mean and median values of each numeric
#Imputation of missing value with median
median_ozone = data["Ozone"].median()
mean_ozone = data["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[20]:


# Replace the Ozone missing values with median value
data['Ozone'] = data['Ozone'].fillna(median_ozone)
data.isnull().sum()


# In[21]:


# Replace the Ozone missing values with median value
data['Ozone'] = data['Ozone'].fillna(mean_ozone)
data.isnull().sum()


# In[ ]:




