#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr


# In[5]:


user1 = np.array([4, 5, 2, 3, 4])
user2 = np.array([5, 3, 2, 4, 5])


# In[6]:


cosine_similarity = 1 - cosine(user1, user2)
print(f"Cosine Similarity: {cosine_similarity:.4f}")


# In[7]:


pearson_corr, _= pearsonr(user1, user2)
print(f"Pearsonr Correlation Similarity: {pearson_corr:.4f}")


# In[9]:


euclidean_distance = euclidean(user1, user2)
euclidean_similarity = 1/ (1 + euclidean_distance)
print(f"Euclidean Distance Similarity: {euclidean_similarity:.4f}")


# In[ ]:


def compute_similarity(df):


# In[13]:


import pandas as pd
users = ['Raju', 'John', 'Ramya', 'kishore']
ratings = [
    [5, 4, 3, 2, 1],  
    [4, 5, 4, 3, 2],  
    [3, 4, 5, 2, 3],  
    [2, 3, 4, 5, 4]   
]
df = pd.DataFrame(ratings, index=users, columns=["Bahuballi", "Mufasa", "Interstellar", "RRR", "Mrs"])
df


# In[ ]:




