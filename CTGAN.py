#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sdv


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df1 = pd.read_csv("C:/Users/AVI BHANDARI/Downloads/network_dataset_labeled.csv")
df1.head()


# In[4]:


df1 = df1.drop('timestamp', axis=1)


# In[5]:


from sdv.metadata import SingleTableMetadata


# In[6]:


from sdv.single_table import CTGANSynthesizer


# In[7]:


metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df1)

# Initialize the synthesizer with metadata
model = CTGANSynthesizer(metadata)

# Fit the model to your data
model.fit(df1)

# Generate synthetic data
synthetic_data = model.sample(1000)
print(synthetic_data.head())


# In[8]:


from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import pandas as pd

class DataSynthesizer:
    def __init__(self, data: pd.DataFrame, epcohs: int = 1000):
        
        self.data = data
        self.epcohs = epcohs
        self.metadata = None
        self.model = None
        self.synthetic_data = None
    
    def _initialize_metadata(self):
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data=self.data)
    
    def _initialize_model(self):
        self.model = CTGANSynthesizer(metadata=self.metadata)
    
    def fit(self):
        if self.model is None:
            self._initialize_metadata()
            self._initialize_model()
        self.model.fit(self.data)
    
    def generate_synthetic_data(self):
        self.synthetic_data = self.model.sample(self.epcohs)
        return self.synthetic_data
   


synthesizer = DataSynthesizer(data=df1)

# Fit the model
synthesizer.fit()

# Generate synthetic data
synthetic_data = synthesizer.generate_synthetic_data()


# In[9]:


synthetic_data.head(10)


# In[10]:


synthetic_data.to_csv("C:/Users/AVI BHANDARI/Downloads/anomaly_detection.csv", index=False)


# In[12]:


df2 = df1[df1['anomaly']==1]


# In[13]:


synthesizer = DataSynthesizer(data=df2)

# Fit the model
synthesizer.fit()

# Generate synthetic data
synthetic_data = synthesizer.generate_synthetic_data()


# In[14]:


synthetic_data.head(10)


# In[15]:


synthetic_data.to_csv("C:/Users/AVI BHANDARI/Downloads/anomaly_seperate_dataset.csv", index=False)


# In[17]:


df3 = pd.concat([df1, synthetic_data], axis=0)


# In[18]:


df3.to_csv("C:/Users/AVI BHANDARI/Downloads/anomaly_combined_dataset.csv", index=False)


# In[ ]:




