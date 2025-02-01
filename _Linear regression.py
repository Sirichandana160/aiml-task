#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[4]:


data1 = pd.read_csv("NewspaperData.csv")
data1.head()


# EDA

# In[5]:


data1.info()


# In[6]:


data1.isnull().sum()


# In[7]:


data1.describe()


# In[9]:


import matplotlib.pyplot as plt

plt.figure(figsize=(6, 3))  # Set figure size
plt.title("Box plot for Daily Sales")  # Set title
plt.boxplot(data1["daily"], vert=False)  # Create horizontal box plot
plt.show()  # Display the plot


# In[10]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# # Observations

# There are no missing values
# The daily column values apperars to be right-skewed
# The sunday column values also appear to be right-skewed
# There are two outliers in both daily column and also in sunday column as observed from the

# Scatter plot and Correlation Strength

# In[12]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[13]:


data1["daily"].corr(data1["sunday"])


# In[14]:


data1[["daily","sunday"]].corr()


# In[15]:


data1.corr(numeric_only=True)


# # Observations on Correlation strength

# The relationship between x (daily) and y (sunday) is seen to be linear as seen from scatter plot
# The correlation is strong and positive with Pearson's correlation coeficient of 0.958154

# #Fit a linear regression model

# In[17]:


# Build regression model
import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[18]:


model1.summary()


# In[ ]:




