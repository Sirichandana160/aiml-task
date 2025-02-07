#!/usr/bin/env python
# coding: utf-8

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import statsmodels.formula.api as smf
# from statsmodels.graphics.regressionplots import influence_plot
# import numpy as np

# In[7]:


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
# In[8]:


cars.info()


# In[9]:


cars.isna().sum()


# # Observations about info(), missing values
-> THere are no missing values
-> There are 81 observations (81 different cars data)
-> The data types of the columns are also relevant and valid
# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'cars' DataFrame is already defined and contains the 'HP' column

fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# Create boxplot
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')  # Remove x-label for the boxplot

# Create histogram with KDE
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns

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

# In[16]:


cars[cars.duplicated()]


# Pair plots and Correlation Corfficients

# In[24]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[25]:


cars.corr()

# HP has positive corelation with sp
# vol and wt has positive corelation
# -Between x and y all the x variable are showing moderate to high correlation strengths,highest being between HP and MPG
# -Therfore this data set qualifies for building a multiple linear regression model to predict MPG
# -among xx colums(x1,x2,x3 and x4),some very high correlation strength observed between SP vs HP,VOL vsWT
# -The high correlation among x columns is not desirable as it might lead to multicollinearity problem
# In[31]:


from sklearn.metrics import mean_squared_error
import numpy as np

# Assuming df1["actual_y1"] and df1["pred_y1"] are already defined
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])

# Print MSE and RMSE (Root Mean Squared Error)
print("MSE :", mse)
print("RMSE :", np.sqrt(mse))


# In[32]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# # Checking for multicollinearity among X-columns using VIF method

# In[35]:


import pandas as pd
import statsmodels.formula.api as smfO.DataFrame(d1)  
Vif_frame


# # Observations for VIF values:

# + The ideal range of VIF values shall be between 0 to 10.However sightly values can be tolerated
# + As seen from the very high VIF values for VOL and WT,it is clear that they are prone to multicollinearity problem
# + Hence is decided to drop one of the columns (either VOL and WT) to overcome the multicollinearity
# + It is decided to drop WT and retain VOL column in further models

# In[50]:


cars1 = cars.drop("WT", axis=1)
cars1.head()


# In[47]:


import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()
model2.summary()

