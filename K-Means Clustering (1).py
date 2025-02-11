#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[8]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[9]:


# Display data1 missinmg values count in each column using isnull().sum()
Univ.isnull().sum()


# In[10]:


Univ1 = Univ.iloc[:,1:]
Univ1


# In[11]:


cols = Univ1.columns
cols


# In[6]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
cols = Univ1.columns
Univ1.drop_duplicates(keep='first', inplace=True)
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1), columns=cols)
scaled_Univ_df.to_csv('scaled_university.csv', index=False)
print(scaled_Univ_df)


# In[13]:


from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[14]:


clusters_new.labels_


# In[15]:


set(clusters_new.labels_)


# In[16]:


Univ['clusterid_new'] = clusters_new.labels_


# In[17]:


Univ


# In[18]:


Univ.sort_values(by = "clusterid_new")


# In[19]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# # Observations:

# + Cluster 2 appears to be the top rated universities cluster as the cut off score,Top10,SFRatio parameter mean values are highest
# + Cluster 1 appears to occupy the middle level rated universities
# + cluster 0 comes as the lower level rated universities

# In[23]:


Univ[Univ['clusterid_new']==0]


# # Finding optimal K value using elbow plot

# In[26]:


wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i,random_state=0 )
    kmeans.fit(scaled_Univ_df)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')_new
plt.show()


# In[ ]:




