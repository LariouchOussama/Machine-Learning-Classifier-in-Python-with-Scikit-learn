#!/usr/bin/env python
# coding: utf-8

# In[21]:


import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve


# In[12]:


# Load dataset
data = load_breast_cancer()


# In[13]:


# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']


# In[14]:


# Look at our data
print(label_names)
print(labels[0])
print(feature_names[0])
print(features[0])


# In[15]:


# Split our data
train, test, train_labels, test_labels = train_test_split(features,
labels,
test_size=0.33,
random_state=42)


# In[17]:


#Building and Evaluating the Model
# Initialize our classifier
gnb = GaussianNB()
# Train our classifier
model = gnb.fit(train, train_labels)


# In[18]:


# Make predictions
preds = gnb.predict(test)
print(preds)


# In[20]:


# Evaluate accuracy
print(accuracy_score(test_labels, preds))

