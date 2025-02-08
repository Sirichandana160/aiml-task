#!/usr/bin/env python
# coding: utf-8

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import statsmodels.formula.api as smf
# from statsmodels.graphics.regressionplots import influence_plot
# import numpy as np

# In[4]:


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
# In[5]:


cars.info()


# In[6]:


cars.isna().sum()


# # Observations about info(), missing values
-> THere are no missing values
-> There are 81 observations (81 different cars data)
-> The data types of the columns are also relevant and valid
# In[7]:


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


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout
plt.show()


# In[12]:


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

# In[13]:


cars[cars.duplicated()]


# Pair plots and Correlation Corfficients

# In[18]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[19]:


cars.corr()


# In[40]:


import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()

model1.summary()


# In[44]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()



# In[ ]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# # Checking for multicollinearity among X-columns using VIF method

# In[ ]:


import pandas as pd
import statsmodels.formula.api as smfO.DataFrame(d1)  
Vif_frame


# In[41]:


import statsmodels.formula.api as smf

# Assuming 'cars2' is your DataFrame, and 'MPG', 'VOL', 'SP', 'HP' are columns in that DataFrame
model2 = smf.ols('MPG ~ VOL + SP + HP', data=cars2).fit()

# Display the summary of the model
model2.summary()


# # Observations for VIF values:

# + The ideal range of VIF values shall be between 0 to 10.However sightly values can be tolerated
# + As seen from the very high VIF values for VOL and WT,it is clear that they are prone to multicollinearity problem
# + Hence is decided to drop one of the columns (either VOL and WT) to overcome the multicollinearity
# + It is decided to drop WT and retain VOL column in further models

# #### Leverage (Hat Values):
# Leverage values diagnose if a data point has an extreme value in terms of the independent variables. A point with high leverage has a great ability to influence the regression line. The threshold for considering a point as having high leverage is typically set at 3(k+1)/n, where k is the number of predictors and n is the sample size.

# In[21]:


# Define variables and assign values
k = 3 
n = 81
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# In[42]:


import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot

# Assuming model1 is a fitted regression model using statsmodels
# For example: model1 = sm.OLS(y, X).fit()

# Create influence plot
influence_plot(model1, alpha=.05)

# Define leverage cutoff (example: typically 2*(k/n) where k is the number of predictors and n is the number of data points)
leverage_cutoff = 2 * (len(model1.model.exog[0]) / len(model1.model.endog))

# Create range for the y-values (influence plot range)
y = [i for i in range(-2, 8)]

# Define x-values (to align with the leverage_cutoff)
x = [leverage_cutoff for i in range(10)]

# Plot the red '+' markers
plt.plot(x, y, 'r+')

# Show the plot
plt.show()


# In[27]:


from statsmodels.graphics.regressionplots import influence_plot


# In[30]:


cars[cars.index.isin([65,70,76,78,79,80])]


# In[37]:


cars2=cars.drop(cars.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)
cars2


# In[39]:


import statsmodels.formula.api as smf

# Assuming 'cars2' is your DataFrame, and 'MPG', 'VOL', 'SP', 'HP' are columns in that DataFrame
model3 = smf.ols('MPG ~ VOL + SP + HP', data=cars2).fit()

# Display the summary of the model
model3.summary()


# In[43]:


df3 = pd.DataFrame()
df3["actual_y3"]=cars2["MPG"]
df3.head()


# In[49]:


pred_y3 = model3.predict(cars2[['VOL', 'SP', 'HP']])

# Add the predictions to df3
df3["pred_y3"] = pred_y3

# Show the first few rows of df3
df3.head()


# In[54]:


from sklearn.metrics import mean_squared_error

# Assuming df3 has the actual and predicted values
mse = mean_squared_error(df3["actual_y3"], df3["pred_y3"])

# Print the MSE
print("MSE:", mse)


# In[57]:


import pandas as pd

# Create a dictionary with the comparison metrics
comparison_data = {
    'Metric': ['R-squared', 'Adj. R-squared', 'MSE', 'RMSE'],
    'Model 1': [0.771, 0.758, 18.89, 4.34],
    'Model 2': [0.770, 0.761, 18.91, 4.34],
    'Model 3': [0.885, 0.880, 8.68, 2.94]
}

# Convert the dictionary into a DataFrame
comparison_df = pd.DataFrame(comparison_data)

# Display the DataFrame
print(comparison_df)


# In[ ]:




