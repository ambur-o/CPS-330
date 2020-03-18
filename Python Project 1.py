#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("/Users/amberowens/Documents/My Documents/school/Python"))


# In[7]:


data = pd.read_excel("/Users/amberowens/Documents/My Documents/school/Python/top50.xlsx")
data.head(10)


# In[23]:


print("Mean value for danceability:", data['Danceability'].mean())
sns.distplot(data['Danceability'])
plt.show()


# In[24]:


print("Mean value for energy:", data['Energy'].mean())
sns.distplot(data['Energy'])
plt.show()


# In[22]:


print("Mean value for energy:")
sns.countplot(y = data['Genre'],order = data['Genre'].value_counts().index)
plt.show()


# In[21]:


sns.countplot(y = data['Artist.Name'],order = data['Artist.Name'].value_counts().index )
plt.show()


# In[24]:


grouped_data = data.groupby(["Artist.Name", "Genre"])["Track.Name"].aggregate("count").reset_index()
grouped_data = grouped_data.pivot('Artist.Name', 'Genre', 'Track.Name')

plt.figure(figsize=(12,8))
sns.heatmap(grouped_data)
plt.title("Frequency of Artist Vs Genre")
plt.show()


# In[29]:


sns.distplot(y = data['Artist.Name'],data['Popularity'].mean())


# In[ ]:




