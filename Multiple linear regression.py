#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[5]:


cars = pd.read_csv("cars.csv")
cars.head()


# DESCRIPTION OF COLUMNS
# MPG:MIlage of the cars(mile per Gallon) (this is Y_column to be predicted)
# HP:Horse power of the car(X1 column)
# VOL:Volume of the car(size)(X2 coumn
#      
