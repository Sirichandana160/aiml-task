#!/usr/bin/env python
# coding: utf-8

# In[35]:


get_ipython().system('pip install mlxtend')


# In[36]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[37]:


titanic = pd.read_csv("Titanic.csv")
titanic


# In[38]:


titanic.info()


# # Observations:

# + Class: Could represent the class or category of an individual (e.g., passenger class on a ship, or a general classification).
# + Gender: Likely represents the gender of individuals (male/female).
# + Age: This should represent the age of individuals but might need cleaning or conversion if it contains non-numeric data.
# + Survived: This is often a binary variable in datasets like Titanic survivors (where 1 could indicate survival and 0 could indicate death). It might be helpful to convert this into an integer or boolean format.

# In[29]:


titanic.isnull().sum()


# In[24]:


titanic.describe()


# #### Observations
# - All columns are object data type and categorical in nature
# - There are no null values
# - As the columns are categorical, we can adopt one-hot-encoding

# In[7]:


titanic['Class'].value_counts()


# In[8]:


import matplotlib.pyplot as plt
counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# In[9]:


import matplotlib.pyplot as plt
counts = titanic['Age'].value_counts()
plt.bar(counts.index, counts.values)


# In[10]:


import matplotlib.pyplot as plt
counts = titanic['Gender'].value_counts()
plt.bar(counts.index, counts.values)


# In[11]:


import matplotlib.pyplot as plt
counts = titanic['Survived'].value_counts()
plt.bar(counts.index, counts.values)


# In[39]:


df = pd.get_dummies(titanic,dtype=int)
df.head()


# In[40]:


df.info()


# Apriori Algorithm

# In[41]:


frequent_itemsets = apriori(df, min_support = 0.05,use_colnames=True,max_len=None)
frequent_itemsets


# In[33]:


frequent_itemsets.info()


# In[43]:


rules = association_rules(frequent_itemsets,metric='lift',min_threshold=1.0)
rules


# In[42]:


rules.sort_values(by='lift', ascending=True)


# In[44]:


frequent_itemsets.iloc[62,1]


# In[45]:


rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[ ]:




