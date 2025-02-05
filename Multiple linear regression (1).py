#!/usr/bin/env python
# coding: utf-8

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import statsmodels.formula.api as smf
# from statsmodels.graphics.regressionplots import influence_plot
# import numpy as np

# In[2]:


cars = pd.read_csv("cars.csv")
cars.head()


# DESCRIPTION OF COLUMNS
# MPG:MIlage of the cars(mile per Gallon) (this is Y_column to be predicted)
# HP:Horse power of the car(X1 column)
# VOL:Volume of the car(size)(X2 coumn
#      

# Assumptions in Multilnear Regression
1.Linearity:The relationship between the predictors(X) and the response (Y) is linear
2.Independence:Observatons are independent of each other.
3.Homoscedasticity:The residuals (y/ - Y_hat) exhibit constant variance at all levels of the distributed.
4.Normal Distrbution of Errors:The residuals of the model are normally distributed.
5.No Multicollinearity: The independent variable variable should not be highly correlated with each other.    
# In[6]:


cars.info()


# In[7]:


cars.isna().sum()


# # Observations about info(), missing values
-> THere are no missing values
-> There are 81 observations (81 different cars data)
-> The data types of the columns are also relevant and valid
# In[10]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# In[11]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# In[14]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# In[15]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# In[16]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# Obseravations from boxplot and histograms
-> There are some extreme values (outliers) observed in towards the right tail of SP and HP distributions.
-> In VOL and WT columns a few outliers are observed in both tails of their distributions.
-> The extreme values of cars data may have come from the specially designed nature of cars.
-> As this is multi-dimensional data the outliers with respect to spatial dimensions may have to be considered while building the regression model.
# # Checking for duplicated rows

# In[19]:


cars[cars.duplicated()]


# Pair plots and Correlation Corfficients

# In[20]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[ ]:




