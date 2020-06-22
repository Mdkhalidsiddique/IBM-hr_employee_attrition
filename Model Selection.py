#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pandas.util.testing as tm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
plt.style.use('seaborn')

#model developemnt libraries
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2

from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv("IBM_hr_data_attrition.csv")
df.head()


# In[3]:



#Get list of columns in the dataset
df.columns


# In[4]:


#Dropping columns (intution)
columns = ['DailyRate', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 'HourlyRate', 'MonthlyRate',
        'Over18', 'RelationshipSatisfaction', 'StandardHours']


# In[5]:


df.drop(columns, inplace=True, axis=1)


# In[6]:


#Find number of missing values in every feature
df.isnull().sum()


# In[7]:


#Columns with string values
categorical_column = ['Attrition', 'BusinessTravel', 'Department',
                      'Gender', 'JobRole', 'MaritalStatus', 'OverTime']


# In[8]:


#Deep copy the original data
data_encoded = df.copy(deep=True)
#Use Scikit-learn label encoding to encode character data
lab_enc = preprocessing.LabelEncoder()
for col in categorical_column:
        data_encoded[col] = lab_enc.fit_transform(df[col])
        le_name_mapping = dict(zip(lab_enc.classes_, lab_enc.transform(lab_enc.classes_)))
        print('Feature', col)
        print('mapping', le_name_mapping)


# In[9]:


data_encoded['Attrition'].value_counts()


# imbalanced data

# In[10]:


data_correlation = data_encoded.corr()


# In[11]:


plt.rcParams["figure.figsize"] = [15,10]
sns.heatmap(data_correlation,xticklabels=data_correlation.columns,yticklabels=data_correlation.columns)


# In[12]:


#Viewing the analysis obtained above 
data_corr_filtered = df[['MonthlyIncome', 'TotalWorkingYears', 'Age', 'MaritalStatus', 'StockOptionLevel',
                      'JobLevel']]
correlation = data_corr_filtered.corr()
plt.rcParams["figure.figsize"] = [20,10]
sns.heatmap(correlation,xticklabels=data_corr_filtered.columns,yticklabels=data_corr_filtered.columns)


# In[13]:


input_data = data_encoded.drop(['Attrition'], axis=1)


# In[14]:


input_data.head()


# In[15]:


target_data = data_encoded[['Attrition']]


# In[16]:


len(input_data.columns)


# In[17]:


input_data.columns


# In[18]:


col_values = list(input_data.columns.values)


# In[19]:


#gives top 10 features having maximum mutual information value
feature_scores = mutual_info_classif(input_data, target_data)
for score, fname in sorted(zip(feature_scores, col_values), reverse=True)[:10]:
    print(fname, score)


# In[20]:



#gives top 10 features having maximum chi-square value
feature_scores = chi2(input_data, target_data)[0]
for score, fname in sorted(zip(feature_scores, col_values), reverse=True)[:10]:
    print(fname, score)


# In[21]:



#column selection based on feature selection 
data_selected = df[['MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 
                      'YearsWithCurrManager', 'Age', 'OverTime', 'DistanceFromHome', 'StockOptionLevel',
                      'JobLevel', 'JobRole', 'WorkLifeBalance', 'Gender', 'Attrition']]


# In[22]:


data_selected.head()


# In[23]:


#encoding labels
data_selected.loc[data_selected.Attrition == 'No', 'Attrition'] = 0
data_selected.loc[data_selected.Attrition == 'Yes', 'Attrition'] = 1


# In[25]:


le = preprocessing.LabelEncoder()
data_selected.JobRole = le.fit_transform(data_selected.JobRole)
data_selected.OverTime = le.fit_transform(data_selected.OverTime)
data_selected.Gender = le.fit_transform(data_selected.Gender)


# In[44]:


X= data_selected.drop(['Attrition'], axis=1)
y = data_selected[['Attrition']]


# In[29]:


X.head()


# In[30]:


Y.head()


# In[36]:


from sklearn.model_selection import train_test_split


# In[59]:


from sklearn.datasets import make_classification
X,y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)


# In[47]:


clf = AdaBoostClassifier(n_estimators=100, random_state=0)


# In[48]:


clf.fit(X, y)


# In[49]:


clf.score(X, y)

