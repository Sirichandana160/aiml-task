#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[5]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[10]:


# Display data1 missinmg values count in each column using isnull().sum()
Univ.isnull().sum()


# In[9]:


Univ1 = Univ.iloc[:,1:]
Univ1


# In[16]:


cols = Univ1.columns
cols


# In[19]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
cols = Univ1.columns
Univ1.drop_duplicates(keep='first', inplace=True)
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1), columns=cols)
scaled_Univ_df.to_csv('scaled_university.csv', index=False)
print(scaled_Univ_df)


# In[ ]:




