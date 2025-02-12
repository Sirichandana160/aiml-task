#!/usr/bin/env python
# coding: utf-8

# ### k-Means Clustering
# K-means clustering is a popular method used in data analysis to partition a dataset into K distinct, non-overlapping clusters. The goal is to group data points into clusters such that the points in each cluster are similar to each other while being different from points in other clusters. Here's a breakdown of how it works:
# 
# 1. **Initialization**: K initial "centroids" are chosen at random from the dataset, where K is the desired number of clusters.
# 
# 2. **Assignment**: Each data point is assigned to the nearest centroid, and thus grouped into a cluster. The "nearest" is typically determined by the distance between a point and a centroid, with Euclidean distance being the most common metric.
# 
# 3. **Update**: Once all data points are assigned to clusters, the centroids of these clusters are recalculated. This is typically done by taking the mean of all data points in each cluster.
# 
# 4. **Iteration**: The assignment and update steps are repeated iteratively until no further changes occur in the centroids, or the changes are minimal. This means that the clusters have become stable and further iterations will not significantly alter the composition of the clusters.
# 
# ### Mathematical formulation
# K-means clustering is used in a wide range of applications, including market segmentation, computer vision, geostatistics, and astronomy, among others. It's particularly valued for its simplicity and speed, but it also has some limitations, such as the need to specify K in advance, sensitivity to the initial choice of centroids, and difficulty in clustering data of varying sizes and density.
# 
# The objective function of k-means clustering, which is minimized, can be described mathematically by the following equation
# $$
# J = \sum_{i=1}^{k} \sum_{\mathbf{x} \in S_i} \|\mathbf{x} - \mathbf{\mu}_i\|^2
# $$
# 
# J is the objective function that needs to be minimized, representing the total within-cluster variance.
# 
# -k is the number of clusters.
# 
# -𝑆𝑖 is the set of all data points
# 
# -x assigned to the i-th cluster
# 
# -𝜇𝑖 is the centroid of the i-th cluster
# 
# -‖𝐱−𝜇𝑖‖^2 represents the squared Euclidean distance between a data point and the centroid 
# of its assigned cluster
# 
# 

# ### Distance Metrics 
# There are several distance metrics commonly used in clustering algorithms. Here are some of the most common ones along with their formulas:
# 
# 1. **Euclidean Distance**:
#    - Formula: $$ \sqrt{\sum_{i=1}^{n} (x_{i} - y_{i})^2}$$ 
#    - Description: Measures the straight-line distance between two points in Euclidean space.
# 
# 2. **Manhattan Distance** (City Block or L1 distance):
#    - Formula: $$ \sum_{i=1}^{n} |x_{i} - y_{i}|$$
#    - Description: Measures the sum of the absolute differences between the coordinates of the points.
# 
# 3. **Chebyshev Distance** (Chessboard or L∞ distance):
#    - Formula: $$\max_{i}(|x_{i} - y_{i}|)$$
#    - Description: Measures the maximum absolute difference between the coordinates of the points along any dimension.
# 
# 4. **Minkowski Distance**:
#    - Formula: $$\left( \sum_{i=1}^{n} |x_{i} - y_{i}|^p \right)^{\frac{1}{p}}$$
#    - Description: Generalized distance metric that includes Manhattan distance (\(p=1\)) and Euclidean distance (\(p=2\)) as special cases.
# 
# 5. **Cosine Similarity**:
#    - Formula: $$ \frac{\mathbf{X} \cdot \mathbf{Y}}{\|\mathbf{X}\| \|\mathbf{Y}\|} $$
#    - Description: Measures the cosine of the angle between two vectors. It is often used for text clustering.
# 
# 6. **Jaccard Distance**:
#    - Formula: $$ 1 - \frac{|A \cap B|}{|A \cup B|}$$
#    - Description: Measures dissimilarity between two sets. It is often used for binary data.
# 
# 7. **Hamming Distance**:
#    - Formula: Number of positions at which the corresponding elements are different between two binary vectors.
#    - Description: Measures the minimum number of substitutions required to change one binary vector into the other.
# 
# 8. **Mahalanobis Distance**:
#    - Formula: $$\sqrt{(\mathbf{X} - \mathbf{Y})^T \mathbf{S}^{-1} (\mathbf{X} - \mathbf{Y})}$$
#    - Description: Measures the distance between a point and a distribution, taking into account the covariance of the data.
# 
# These are just a few examples of distance metrics used in clustering. The choice of distance metric depends on the nature of the data and the clustering algorithm being used.

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# **Clustering- Divide the universities in to groups (Clusters)**

# In[5]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[6]:


Univ.info()


# In[7]:


Univ.isna().sum()


# In[8]:


Univ.describe()


# #### Standardization of the data

# In[10]:


# Read all numeric columns in to Univ1
Univ1 = Univ.iloc[:,1:]
Univ1


# In[11]:


cols = Univ1.columns


# In[12]:


# Standardisation function 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols )
scaled_Univ_df


# In[13]:


scaled_Univ_df.info()


# In[14]:


# How to find optimum number of  cluster
#The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion:


# #### Finding optimal k value using elbow plot

# In[16]:


wcss = []
for i in range(1, 20):
    
    kmeans = KMeans(n_clusters=i,random_state=0 )
    kmeans.fit(scaled_Univ_df)
    #kmeans.fit(Univ1)
    wcss.append(kmeans.inertia_)
print(wcss)    
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[17]:


# Build 3 clusters using KMeans Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0) # Specify 3 clusters
clusters_new.fit(scaled_Univ_df)


# In[18]:


# print the cluster labels
clusters_new.labels_


# In[19]:


set(clusters_new.labels_)


# In[20]:


#Assign clusters to the Univ data set
Univ['clusterid_new'] = clusters_new.labels_


# In[21]:


Univ


# In[22]:


# Univ.to_csv("university_clusters")


# In[23]:


Univ.sort_values(by = "clusterid_new")


# In[24]:


Univ[Univ['clusterid_new']==1]


# In[25]:


#these are standardized values.
clusters_new.cluster_centers_


# In[26]:


Univ[Univ['clusterid_new']==1]


# In[27]:


# Use groupby() to find aggregated (mean) values in each cluster
Univ.iloc[:,1:].groupby("clusterid_new").mean()


# #### Observations:
# - Custer 2 appears to be the top rated universities cluster as the cut off score, Top10, SFRatio parameter mean values are highest
# - Cluster 1 appears to occupy the middle level rated universities
# - Cluster 0 comes as the lower level rated universities

# In[29]:


Univ[Univ['clusterid_new']==0]


# ### Silhoutte Score- a measure of cluster quality
# 
# ![image.png](attachment:b7698ab0-2fbb-4841-b696-6a34bb634ee7.png)!
# 
# **S(i) is the silhouette coefficient of the data point i.**
# 
# **a(i) is the average distance between i and all the other data points in the cluster to which i belongs.**
# 
# **b(i) is the average distance from i to all clusters to which i does not belong.**
# 
# **Mean value of S(i) is calculated**
# 
# **The silhouette score falls within the range [-1, 1].**
# 
# **The silhouette score of 1 means that the clusters are very dense and nicely separated. The score of 0 means that clusters are overlapping. The score of less than 0 means that data belonging to clusters may be wrong/incorrect.**
# 

# In[31]:


# Quality of clusters is expressed in terms of Silhoutte score

from sklearn.metrics import silhouette_score
score =silhouette_score(scaled_Univ_df, clusters_new.labels_ , metric='euclidean')
score


# In[ ]:




