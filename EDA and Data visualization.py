#!/usr/bin/env python
# coding: utf-8

# # Description: 
# 

# Uncover the factors that lead to employee attrition and explore important questions such as ‘show me a breakdown of distance from home by job role and attrition’ or ‘compare average monthly income by education and attrition’. This is a fictional data set created by IBM data scientists.
# 
# # Tasks:
# * Finds out factors which affects emp attition 
# * Find out factors on which affecting factors are depends. 
# * Use Statistical tools to prove inference. 
# 

# In[104]:


import pandas as pd
import pandas.util.testing as tm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
plt.style.use('seaborn')

import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
from plotly import tools
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[2]:


data = pd.read_csv("IBM_hr_data_attrition.csv")
data.head()


# In[3]:


#Shape(size of the dataset)
data.shape


# In[4]:


data.columns


# In[5]:


#Dropping some unnecassary features
data = data.drop(['DailyRate','EmployeeNumber','EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','MonthlyRate','Over18','PerformanceRating','RelationshipSatisfaction','StockOptionLevel'], axis =1)
data.head()


# Dataset has 1470 rows and 35 columns

# In[6]:


data.info()


#  No null values in the dataset that's great

# In[7]:


from pandas_profiling import ProfileReport


# In[8]:


#report = ProfileReport(data, title = 'Pandas_profiling', explorative = True)


# In[9]:


#report.to_file("widgets.html")


# In[10]:


data.describe()


# In[11]:


print(data.nunique())


# In[12]:


#Educational qualification
data['EducationField'].value_counts().plot.barh(title='Education Feild')


# The maaximum employees have 3rd category of education, This might be bachelors.
# 

# In[13]:


#Educational qualification
data['Education'].value_counts().plot.pie(title='Education Type')


# In[ ]:





# In[ ]:





# In[14]:


data['Attrition'] = data['Attrition'].map({ 
    'No': 0, 
    'Yes': 1
}.get)
data.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


data['Gender'] = data['Gender'].map({ 
    'Male': 0, 
    'Female': 1
}.get)
data.head()


# In[16]:


#Age range
sns.set(font_scale=1.5) 
from scipy.stats import norm
fig, ax = plt.subplots(figsize=(15, 9))
sns.distplot(a=data.Age, kde=False, fit=norm)


# Age is normally distrubuted, most of the employees are aged around 30 - 40

# In[17]:


#education vs jobLevel
sns.barplot(x="Education", y="JobLevel", data=data);


# In[18]:


#educationfeild vs jobLevel
plt.figure(figsize=(15, 7))
sns.barplot(x="EducationField", y="JobLevel", data=data);


# As per above plot  most of employees are from marketing field

# In[19]:


#education vs jobLevel
sns.lineplot(x="Education", y="Attrition", data=data);


# In[20]:


#sns.lineplot(x = 'Salary', y = 'Education', data = data)


# * As per the above plot the education level affects the job level
# * let me see if the job level affects the attrition
#  

# In[21]:


#education vs jobLevel
sns.lineplot(x="JobLevel", y="Attrition", data=data);


# By seeing the above plot it is clear that joblevel affects the attrition
# * Job level1 has maximum attrition
# * Job level5 has minimum attrition

# In[22]:


table = pd.pivot_table(data=data,index='Department',values='EmployeeCount',aggfunc=np.sum)
table


# * So R and D has maximum number of employees
# * now let me see does the department affect the attrition
# 

# In[23]:


#attrition vs dept
sns.lineplot(x="Department", y="Attrition", data=data);


# * So here R and D has less attrition and sales department has maximum attrition
# * Let me see the reason for high attrition in Sales and HR dept
# 

# In[24]:


#attrition vs dept
plt.figure(figsize=(15, 7))
sns.barplot(x="DistanceFromHome", y="Attrition",hue = 'Department' ,data=data);


# In[25]:


data['MonthlyIncome'].plot.hist(orientation='horizontal', cumulative=True)


# So here the distance may be the reason for attrition but not for all

# In[26]:


data["MonthlyIncome"] = data["MonthlyIncome"].astype(int,copy=True)


# In[27]:


bins = np.linspace(min(data["MonthlyIncome"]),max(data["MonthlyIncome"]),4)
bins


# In[28]:


group_names = ['Low', 'Medium', 'High']


# In[29]:


data['Salary'] = pd.cut(data['MonthlyIncome'], bins, labels=group_names ,include_lowest = True)
data[['MonthlyIncome','Salary']].head(20)


# In[30]:


import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, data["Salary"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("Salary")
plt.pyplot.ylabel("count")
plt.pyplot.title("Salary")


# In[31]:


#Salary vs attrition
sns.lineplot(x="Salary", y="Attrition", data=data);


# So Salary is a factor that highly affects the attrition

# * now let me see the salaries for the departments

# In[32]:


#data['Department'] = data['Department'].map({ 
    #'Sales': 1, 
    #'Research & Development': 2, 'Human Resources': 3
#}.get)
#data.head()


# In[33]:


#dept vs income
sns.barplot(x="Department", y="MonthlyIncome", data=data);


# even though R and D has less income compared to Sales and Hr dept attrititon is less

# In[36]:


plt.figure(figsize=(10,10))
sns.catplot(x = 'Department', y = 'YearsSinceLastPromotion', data = data)


# So promotion may be the possible reason for attrition as hr dept has less promotion and more attrition

# In[37]:


sns.countplot(y='MaritalStatus', data=data)


# In[38]:


plt.figure(figsize=(15, 7))
sns.barplot(x = 'TotalWorkingYears', y = 'Attrition', data = data)


# Looks like some outliers are present, let me check

# In[39]:


sns.boxplot(data = data['TotalWorkingYears'])


# In[40]:


print(data['TotalWorkingYears'].quantile(0.5))
print(data['TotalWorkingYears'].quantile(0.95))


# In[41]:


data["TotalWorkingYears"] = np.where(data["TotalWorkingYears"] <0.0, 0.0,data['TotalWorkingYears'])
data["TotalWorkingYears"] = np.where(data["TotalWorkingYears"] >30.0, 30.0,data['TotalWorkingYears'])
sns.boxplot(data = data['TotalWorkingYears'])


# In[42]:


#Totalyears(Experience) vs Attrition
plt.figure(figsize=(15, 7))
sns.barplot(x = 'TotalWorkingYears', y = 'Attrition', data = data)


# As per the above plot, employees working for longer period in the company are less likely to attrite. The reason for this may be due to high salary

# In[43]:


#Salary vs Years
plt.figure(figsize=(15, 7))
sns.boxplot(x = 'Salary', y = 'TotalWorkingYears', data = data)


# its clear that as the salary increases with the experience, hence the employee are less likely to attrite

# In[44]:


sns.barplot(x = 'Department', y = 'TotalWorkingYears', hue = 'Attrition', data = data)


# In[45]:


import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
from plotly import tools


# In[46]:


Type=data.groupby('Department')['Gender'].agg('count')

values=[Type['Sales'],Type['Research & Development'],Type['Human Resources']]
labels=['Sales','R & D', 'HR']

trace=go.Pie(labels=labels,values=values)
py.iplot([trace])


# In[47]:


data.columns


# In[48]:


data.info()


# # Treating categorical data

# In[49]:


le = preprocessing.LabelEncoder()


# In[50]:


data. BusinessTravel  = le.fit_transform(data. BusinessTravel )
data.Department   = le.fit_transform(data.Department)
data.EducationField = le.fit_transform(data.EducationField)
data.JobRole = le.fit_transform(data.JobRole)
data.MaritalStatus = le.fit_transform(data.MaritalStatus)
data.OverTime = le.fit_transform(data.OverTime)
data.Salary = le.fit_transform(data.Salary)
data.head()


# In[51]:


data.describe()


# In[52]:


list(set(data.dtypes.tolist()))


# In[53]:


data_num = data.select_dtypes(include = ['float64', 'int64', 'int32'])
data_num.head()


# In[54]:


corr = data_num.drop('Attrition', axis=1).corr()
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# In[55]:


data_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations


# * Their are some skewed columns in the dataset, i will use log10 transformation to get normal distribution
# * This will give normal distribution as well as, it will treat outliers

# In[70]:


data['DistanceFromHome'] = np.log10(data['DistanceFromHome'])
data['HourlyRate'] = np.log(data['HourlyRate'])
data['PercentSalaryHike'] = np.log10(data['PercentSalaryHike'])


# 
# 
# # outlier treatment

# In[71]:


data['YearsAtCompany'].describe()


# In[77]:


data['YearsInCurrentRole'].describe()


# In[72]:


sns.boxplot(data['YearsAtCompany'])


# In[73]:


print(data['YearsAtCompany'].quantile(0.5))
print(data['YearsAtCompany'].quantile(0.95))


# In[76]:


data["YearsAtCompany"] = np.where(data["YearsAtCompany"] <5.0, 5.0,data['YearsAtCompany'])
data["YearsAtCompany"] = np.where(data["YearsAtCompany"] >20.0, 20.0,data['YearsAtCompany'])
sns.boxplot(data = data['YearsAtCompany'])


# In[78]:


sns.boxplot(data['YearsInCurrentRole'])


# In[82]:


print(data['YearsInCurrentRole'].quantile(0.5))
print(data['YearsInCurrentRole'].quantile(0.95))


# In[81]:


data["YearsInCurrentRole"] = np.where(data["YearsInCurrentRole"] <5.0, 5.0,data['YearsInCurrentRole'])
data["YearsInCurrentRole"] = np.where(data["YearsInCurrentRole"] >20.0, 20.0,data['YearsInCurrentRole'])
sns.boxplot(data = data['YearsInCurrentRole'])


# In[88]:


data.describe()


# In[89]:


data.columns

