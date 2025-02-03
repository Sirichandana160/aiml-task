#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewspaperData.csv")
data1.head()


# EDA

# In[3]:


data1.info()


# In[4]:


data1.isnull().sum()


# In[5]:


data1.describe()


# In[6]:


import matplotlib.pyplot as plt

plt.figure(figsize=(6, 3))  # Set figure size
plt.title("Box plot for Daily Sales")  # Set title
plt.boxplot(data1["daily"], vert=False)  # Create horizontal box plot
plt.show()  # Display the plot


# In[7]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# # Observations

# There are no missing values
# The daily column values apperars to be right-skewed
# The sunday column values also appear to be right-skewed
# There are two outliers in both daily column and also in sunday column as observed from the

# Scatter plot and Correlation Strength

# In[8]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[9]:


data1["daily"].corr(data1["sunday"])


# In[10]:


data1[["daily","sunday"]].corr()


# In[11]:


data1.corr(numeric_only=True)


# # Observations on Correlation strength

# The relationship between x (daily) and y (sunday) is seen to be linear as seen from scatter plot
# The correlation is strong and positive with Pearson's correlation coeficient of 0.958154

# #Fit a linear regression model

# In[12]:


# Build regression model
import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[13]:


model1.summary()


# # Interpretation

# In[15]:


# plot the scatter plot and overlay the fitted straight line using matplotlib
x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 = 1.33
# predicated response vctor
y_hat = b0 + b1*x

#ploting the regression line
plt.plot(x, y_hat, color = "g")

#putting labels
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# The probaility(p-value) for intercept(beta_0) is 0.707 > 0.05
# Therefore the intercept coefficinet may not be the much significant in prediction
# However the p-value for "daily" (beta_1) is 0.00 < 0.05
# Therefore the beta_1 coefficient is highly significant and is contributint to prediction

# In[16]:


# print the fitted the coefficients (beta_0 and beta_1)
model1.params


# In[25]:


print(f'model t-values:\n{model1.tvalues}\n---------------\nmodel p-values:\n{model1.pvalues}')


# In[26]:


print(f'R-squared: {model1.rsquared}')


# predict for new data point

# In[22]:


newdata=pd.Series([200,300,1500])


# In[23]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[24]:


model1.predict(data_pred)


# In[29]:


pred = model1.predict(data1["daily"])
pred


# In[30]:


data1["Y_hat"] = pred
data1


# In[31]:


data1["residuals"]=data1["sunday"]-data1["Y_hat"]
data1


# In[ ]:




