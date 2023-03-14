#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the Libraries:
import numpy as np 
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pylab
import statsmodels.api as sm
import statistics
from scipy import stats
import sklearn


# # G2M insight for Cab Investment firm:
# Data sets given are all in Csv file format:
# 
# Cab_data :-Details regarding Transaction Ids,travel dates,Cab company,distance travelled,costs and prices
# Customer_ID:-Customer details about gender, their age and Income.
# Transaction_ID:-Transaction information like payment method
# City:-Details about the cities in which cabs operates, number of users and population demographics

# #Study the given data sets:
# Cab_data

# In[2]:


Cabdata=pd.read_csv("D:\DataGlacier\Week2_EDA\Cab_Data.csv",low_memory=False)
Cabdata


# In[3]:


Cabdata.info()


# In[4]:


# View of basic statistical details:


# In[5]:


Cabdata.describe()


# Avg KM Travelled-22.56 miles,Price Charged-$422.74,Cost of each trip-$286.16

# In[6]:


Cabdata.dtypes


# Number of unique Companies,Cities:

# In[7]:


Cabdata['Company'].nunique()


# In[8]:


Cabdata['Company'].unique()


# In[9]:


Cabdata['City'].nunique()


# In[10]:


Cabdata['City'].unique()


# In[11]:


#Check for any null values:


# In[12]:


Cabdata.isnull().sum()


# In[13]:


#Check for any duplicate values:
(Cabdata.duplicated().sum())


# City Data:

# In[14]:


Citydata=pd.read_csv("D:\DataGlacier\Week2_EDA\City.csv",low_memory=False)
Citydata


# In[15]:


Citydata.info()


# The data types for columnns(features) Users and Population are object, we convert them to interger types because we require them for the analysis later.

# In[16]:


Citydata.Population = [Citydata.Population[i].replace(",", "") for i in range(len(Citydata))]
Citydata.Users = [Citydata.Users[i].replace(",", "") for i in range(len(Citydata))]
Citydata.head(2)


# In[17]:


Citydata['Users'] = pd.to_numeric(Citydata['Users'], errors="coerce")
Citydata['Population'] = pd.to_numeric(Citydata['Population'], errors="coerce")


# In[18]:


Citydata.info()


# In[19]:


Citydata.describe()


# Checking for Null,duplicated records:

# In[20]:


(Citydata.duplicated().sum())


# In[21]:


(Citydata.isnull().sum())


# Customer_ID dataset:

# In[22]:


Customerdata=pd.read_csv("D:\DataGlacier\Week2_EDA\Customer_ID.csv",low_memory=False)
Customerdata


# In[23]:


Customerdata.info()


# In[24]:


Customerdata.describe(include='all')


# Checking for any Null and duplicates:

# In[25]:


Customerdata.isnull().sum()


# In[26]:


Customerdata.duplicated().sum()


# Trasaction_ID data set:

# In[27]:


Transactiondata=pd.read_csv("D:\DataGlacier\Week2_EDA\Transaction_ID.csv",low_memory=False)
Transactiondata


# In[28]:


Transactiondata.info()


# In[29]:


Transactiondata.describe(include='all')


# Check the data set for Null or duplicated values:

# In[30]:


Transactiondata.isnull().sum()


# In[31]:


Transactiondata.duplicated().sum()


# Cab data contains feature Transaction ID which holds key to data set Transaction_ID dataset which in turn holds key to connect Customer_ID data details via Customer_ID key hence we merge these data set along with appending the details if City dataset through the feature City.

# In[32]:


Masterdata=Cabdata.merge(Transactiondata,on='Transaction ID').merge(Customerdata,on='Customer ID').merge(Citydata,on='City')
Masterdata


# # Exploratory Data Analysis(EDA)

# Analysis of the entire data set and by each feature

# In[33]:


correlation=Masterdata.corr()
correlation


# In[34]:


heatmapdata = Masterdata[['Transaction ID','Date of Travel','Company','City','KM Travelled','Price Charged','Cost of Trip','Population','Users','Age']]
sns.heatmap(heatmapdata.corr(), annot=True)
plt.show()


# There exists a linear positive correlation between * features KM Travelled, Price Charged, Cost of Trip 
# and * features Population and Users

# In[35]:


#Outlier Detection-Continuous Variables Box Plots


# In[36]:


plt.boxplot('KM Travelled',data=Masterdata)
plt.show()


# In[37]:


plt.boxplot('Price Charged',data=Masterdata)
plt.show()


# In[38]:


plt.boxplot('Cost of Trip',data=Masterdata)
plt.show()


# Clearly, feature(column) 'Price Changed' contains Outliers

# In[39]:


mycolor=['yellow','pink']
Masterdata.Company.value_counts().plot.pie(y='Company', figsize=(5, 5), autopct='%1.0f%%',colors=mycolor)


# Yellow cab has majority coverage with 76% 

# In[40]:


#Number of Transactions occured by each Company


# In[41]:


a1=Masterdata.groupby(['Company'])['Transaction ID'].count()

plt.figure(figsize=(4,9))
plt.title('Number of Transactions ocuured per Company')
sns.countplot(data=Masterdata,x='Company',palette=['pink','yellow'])


# In[42]:


#Company prominence by City:

plt.figure(figsize=(16,9))
plt.title('Cab Pink and Cab Yellow across Cities')
sns.countplot(data=Masterdata,x='City',hue='Company',palette=['pink','yellow'])
plt.xlabel("Cities",fontsize=10)
plt.ylabel("Count", fontsize=10)
plt.xticks(rotation=30, ha='right')


# New York City NY state ,Washington DC,Chicago IL,Los Angeles CA and Boston MA cities have the highest Cab presence and Yellow Cab is of prefered over Pink Cab in most of the cities .
# Pink Cab -San diego,Nashville,Sacremento,Pittsburg

# In[43]:


Masterdata.groupby(['Company','City'])['Customer ID'].count()


# In[44]:


#Date of Travel per company wise


# To analyze the Cab usage during particular periods the Date of Travel feature is further split to Year,Month,Day and DayofWeek

# In[45]:


Masterdata['Date of Travel'] = pd.to_datetime(Masterdata['Date of Travel'])
#Separate Day,Month,Year
Masterdata['Year'] = Masterdata['Date of Travel'].dt.strftime('%Y')
Masterdata['Month'] = Masterdata['Date of Travel'].dt.strftime('%m')
Masterdata['Day'] = Masterdata['Date of Travel'].dt.strftime('%d')
Masterdata['DayofWeek'] = Masterdata['Date of Travel'].dt.strftime('%w')
Masterdata


# In[46]:


plt.figure(figsize=(16,9))
pd.crosstab(Masterdata['Date of Travel'], Masterdata['Company'], normalize = "index")
sns.countplot(x = Masterdata['Date of Travel'], hue = Masterdata['Company'],palette=['pink','yellow'])


# Yellow Cabs seem to have more number of rides during Days of Travel

# In[47]:


plt.figure(figsize=(16,9))
pd.crosstab(Masterdata['Date of Travel'], Masterdata['Company'], normalize = "index")
sns.countplot(x = Masterdata['Year'], hue = Masterdata['Company'],palette=['pink','yellow'])


# Yellow Cab company have outpassed Pink Cab all the years 2016-2018 and highest rides in 2017

# In[48]:


plt.figure(figsize=(16,9))
pd.crosstab(Masterdata['Date of Travel'], Masterdata['Company'], normalize = "index")
sns.countplot(x = Masterdata['Month'], hue = Masterdata['Company'],palette=['pink','yellow'],order=Masterdata['Month'].value_counts().index)


# Cabs are at peak usage during December,October and November months and Yellow cabs are mostly preferred 

# In[49]:


plt.figure(figsize=(16,9))
pd.crosstab(Masterdata['Date of Travel'], Masterdata['Company'], normalize = "index")
sns.countplot(x = Masterdata['Day'], hue = Masterdata['Company'],palette=['pink','yellow'],order=Masterdata['Day'].value_counts().index)


# Cabs rides are higher during the second week of the month

# In[51]:


plt.figure(figsize=(16,9))
pd.crosstab(Masterdata['Date of Travel'], Masterdata['Company'], normalize = "index")
sns.countplot(x = Masterdata['DayofWeek'], hue = Masterdata['Company'],palette=['pink','yellow'],order=Masterdata['DayofWeek'].value_counts().index)


# Cab rides clearly peak during the weekend(6-Sunday,5-Saturday,0-Monday) and Mondays

# In[52]:


#KM Travelled


# In[53]:


plt.figure(figsize=(20,10))
plt.hist(Masterdata['KM Travelled'],bins=50,alpha=0.5, histtype='bar', ec='black')
plt.title('KM Travelled by Cabs',fontsize=24)
plt.xlabel('KM Travelled')
plt.ylabel('Freqency')
plt.show()
         


# Average KM Travelled-22.56 miles with a max of 48 miles and min of 1.9 miles

# In[54]:


#Price Charged


# In[55]:


price=Masterdata.groupby(['Company']).sum()['Price Charged']
price.round(1)


# In[56]:


Masterdata.Payment_Mode.value_counts().plot.pie(y='Payment_Mode', figsize=(5, 5), autopct='%1.0f%%')


# 60% Customers pay for Cab rides via Card mode amd 40% prefer paying by Cash

# In[57]:


#Price Charged by respective Cab companies via Mode of Payment visual:


# In[58]:


modeofpay=Masterdata.groupby(['Company','Payment_Mode']).sum()['Price Charged']
modeofpay=modeofpay.reset_index()


# In[59]:


plt.title('Price Charged via Payment mode')
sns.barplot(data=modeofpay,hue='Company',x='Payment_Mode',y='Price Charged',palette=['pink','yellow'])
plt.show()


# Among the mode of payment Yellow cab company have higher number of customers paying by Card and Cash

# In[60]:


#Profit earned by respective Cab companies in each year from 2016-2018:


# In[61]:


Masterdata['Date of Travel'] = pd.to_datetime(Masterdata['Date of Travel'])
Masterdata['Year'] = Masterdata['Date of Travel'].dt.strftime('%Y')
Masterdata['Month'] = Masterdata['Date of Travel'].dt.strftime('%m')
Masterdata['Day'] = Masterdata['Date of Travel'].dt.strftime('%d')
Masterdata


# In[62]:


profit=Masterdata.groupby(['Company']).sum()['Price Charged']
profit.round(1)


# In[64]:


sns.lineplot(x='Year',y='Profit',hue='Company',data=Masterdata,palette=['pink','yellow'])
plt.xlabel('Years')
plt.ylabel('Profit earned per year')
plt.show()


# Yellow Cab have higher profit over Pink Cab but both companies see a dip in profit after year 2017

# In[65]:


sns.lineplot(x='Month',y='Profit',hue='Company',data=Masterdata,palette=['pink','yellow'])
plt.xlabel('Months')
plt.ylabel('Profit earned per month')
plt.show()


# Yellow Cab company soars profit during June and sees a low dip during August while Pink cab profits are high during month December and lows during June
# Yellow cab has higher profit margin over pink cab company

# In[66]:


#Profit wrt each Cities:


# In[67]:


cityprofit=Masterdata.groupby(['City']).sum()['Profit']
cityprofit.round(1)


# In[68]:


cityprofit.max()


# In[69]:


cityprofit.min()


# The highest Profit grossing City is NEW YORK NY 27617219.6 and the lowest profit wrt City is from PITTSBURGH PA 83197.7

# In[70]:


cities=Masterdata.groupby('City')
cities=cities.Profit.sum()
ind=cities.index
values=cities.values


# In[71]:


figp,axp=plt.subplots(figsize=(16,20))
axp.pie(values,labels=ind, autopct='%1.0f%%',startangle=90)
plt.axis('equal')
plt.show()


# In[72]:


# Customers and Users citywise and company wise:


# In[73]:


users1=Masterdata.groupby(['City']).count()['Users']
#users1=users1.reset_index()

a=users1.index
b=users1.values
users1


# In[74]:


figp,axp=plt.subplots(figsize=(10,20))
axp.pie(b,labels=a, autopct='%1.0f%%',startangle=90)
plt.axis('equal')
plt.show()


# New York NY,Chicago IL,Los Angeles CA and Washington DC contribute to the higher number of users riding cabs

# In[75]:


plt.figure(figsize=(16,9))
pd.crosstab(Masterdata['City'], Masterdata['Users'], normalize = "index")
sns.countplot(x = Masterdata['City'], hue = Masterdata['Users'], order=Masterdata['City'].value_counts().index)
plt.xlabel('Cities')
plt.ylabel('User Count')
plt.xticks(rotation=30, ha='right')


# Number of users using Cab Pink and Cab Yellow:

# In[76]:


compuser=Masterdata.groupby(['Company']).count()['Users']
compuser=compuser.reset_index()
compuser


# In[77]:


plt.title('No. of users using Cab Pink and Yellow')
sns.barplot(data=compuser,x='Company',y='Users',palette=['pink','yellow'])
plt.show()


# Users prefer Yellow cabs

# Users using respective Cab companys by Cities:

# In[78]:


users2=Masterdata.groupby(['Company','City']).count()['Users']
users2=users1.reset_index()
users2


# Gender based analysis:
# 

# In[80]:


Masterdata.Gender.value_counts().plot.pie(y='Gender', figsize=(5, 5), autopct='%1.0f%%')


# In[81]:


gender=Masterdata.groupby(['Company','Gender']).nunique()['Customer ID']
gender=gender.reset_index()
gender


# In[82]:


plt.title('Gender based Company preference')
sns.barplot(data=gender,hue='Company',x='Gender',y='Customer ID',palette=['pink','yellow'])
plt.show()


# Male cab service users are higher compared to Female riders 

# In[83]:


#Gender across cities:


# In[84]:


gender2=Masterdata.groupby(['City','Gender']).nunique()['Customer ID']
gender2=gender2.reset_index()
gender2


# In[102]:


plt.figure(figsize=(16,9))
plt.title('Gender percent across Cities')
sns.barplot(data=gender2,hue='Gender',x='City',y='Customer ID',palette=['pink','cyan'])
plt.xticks(rotation=30, ha='right')
plt.show()


# Average Age of Users:

# In[85]:


age=Masterdata.groupby(['Gender']).mean()['Age']
age


# Average age of Users is 35 years

# In[86]:


plt.figure(figsize=(14,9))
sns.boxplot(x='Gender', y='Age', data=Masterdata, notch=False)
plt.title('Average Age of Male and Female customers', fontsize=14)
plt.show()          


# In[ ]:


#Income


# In[128]:


income=Masterdata.groupby(['Company']).mean()['Income (USD/Month)']
income=income.reset_index()
income


# In[129]:


plt.title('Income range across companies')
sns.barplot(data=income,x='Company',y='Income (USD/Month)',palette=['pink','yellow'])
plt.show()


# Average income of users is $15015.63

# Users of Cab in terms of Population
# 

# In[87]:


user3=(Citydata['Users']/Citydata['Population'])*100
city=Citydata['City']


# In[88]:


plt.figure(figsize=(16,9))
plt.bar(city,user3,edgecolor='black')
plt.xlabel('Population')
plt.ylabel('Users percent')
plt.title('Users of Cab in terms of Population')
plt.xticks(rotation=30, ha='right')
plt.show()


# In terms of population demograhics San Fransisco CA,Boston MA and Washington DC are higher

# # Hypothesis Testing

# Null Hypothesis(H0):There is no statistically significant effect in the population
# Alternative Hypothesis(H1):There is a statistically significant effect in the population

# Test-Hypothesis-1-T-Test
# 
# H0: Age does not have significant effect on the Compnay chosen
# H1:Age does have significant effect on the Compnay chosen

# In[89]:


_,pvalue1=stats.ttest_ind(Masterdata['Age'][Masterdata['Company'] == 'Pink Cab'],
                Masterdata['Age'][Masterdata['Company'] == 'Yellow Cab'],equal_var=True)


# In[90]:


print(pvalue1)


# In[92]:


if (pvalue1 < 0.05):
    print('Accept alternative H1 that Age does have significant effect on the Compnay chosen p-value <0.05')
else: print('Reject alternative H1 and accept Null hypothesis that Age does not have significant effect on the Compnay chosen p-value >0.05')


# Test-Hypothesis-2-T-Test
# Test-Hypothesis-1-T-Test
# 
# H0: Gender does not have significant effect on the Compnay profit
# H1:Gender does have significant effect on the Compnay profit

# In[93]:


_,pvalue2=stats.ttest_ind(Masterdata['Profit'][Masterdata['Gender'] == 'Male'],
                Masterdata['Profit'][Masterdata['Gender'] == 'Female'],equal_var=True)


# In[94]:


print(pvalue2)


# In[95]:


if (pvalue2 < 0.05):
    print('Accept alternative H1 that Gender does have a significant effect on the Compnay profit (p-value <0.05)')
else: print('Reject alternative H1 and accept Null hypothesis H0 that Gender does have a significant effect on the Compnay profit (p-value >0.05)')


# In[96]:


_,pvalue3=stats.ttest_ind(Masterdata['Income (USD/Month)'][Masterdata['Company'] == 'Pink Cab'],
                Masterdata['Income (USD/Month)'][Masterdata['Company'] == 'Yellow Cab'],equal_var=True)


# In[97]:


print(pvalue3)


# In[98]:


if (pvalue3 < 0.05):
    print('Accept alternative H1 that Income does have a significant effect on the Compnay preference (p-value <0.05)')
else: print('Reject alternative H1 and accept Null hypothesis H0 that Income does not have a significant effect on the Compnay preference (p-value >0.05)')


# Test-Hypothesis--T-Test
# 
# H0: Gender does not have significant effect on the Income
# H1:Gender does have significant effect on the Income

# In[99]:


_,pvalue4=stats.ttest_ind(Masterdata['Income (USD/Month)'][Masterdata['Gender']=='Female'],
                           Masterdata['Income (USD/Month)'][Masterdata['Gender']=='Male'],equal_var=True)


# In[100]:


print(pvalue4)


# In[101]:


if (pvalue3 < 0.05):
    print('Accept alternative H1 that Gender does have a significant effect on the Income  (p-value <0.05)')
else: print('Reject alternative H1 and accept Null hypothesis H0 that Gender does not have a significant effect on the Income (p-value >0.05)')

