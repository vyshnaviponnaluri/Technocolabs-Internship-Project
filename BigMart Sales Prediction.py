#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Big Mart Sales Data
#Importing the libraries
#Importing Pandas Library for reading the files
import pandas as pd
#Now reading the data file Train.csv




# In[4]:


train=pd.read_csv("Train.csv")
train.head()

test =pd.read_csv("Test.csv")
test.head()
#Now we will add one more column extra so that we will get to the know from the data which is train and test data
train['source']='train'
test['source']='test'
train.head()
test.head()

#Now we will combine the two datasets
databig_mart=pd.concat([train,test],ignore_index=True)
databig_mart.head()
databig_mart.shape
print (train.shape,test.shape,databig_mart.shape)
#So , now we are having equal no.of.columns in the combined data test also


# In[24]:


#Perform the summary statistics on the entire dataset
databig_mart.describe()


# In[10]:


databig_mart.apply(lambda x:sum(x.isnull()))


# In[ ]:


#From the summary statistics the following points can be infered
#Item_Visibility : Minimum value for the Item Visibility is given as zero,but if there is a product in the store then 
#there will be no scope such as Item Visibility as zero
#Missing Values : Now the two columns Item_weight and Outlet_size are having missing values.


# In[51]:


#Univariate Analysis for each column
# databig_mart['Item_Weight'].describe()
# databig_mart['Item_Weight'].median()
# databig_mart['Item_Weight'].mode()
# databig_mart['Item_Weight'].skew(axis=0)
# databig_mart.boxplot(column='Item_Weight')


# # In[52]:


# #Historgram of Item_Weight Column
# databig_mart.hist(column='Item_Weight')
# #We can infer that Item_weight is slightly right skewed as some of the items may be more weighing


# # In[54]:


# #Item _Visibility
# print(databig_mart['Item_Visibility'].describe())
# print(databig_mart['Item_Visibility'].median())
# print(databig_mart['Item_Visibility'].mode())
# print(databig_mart['Item_Visibility'].skew(axis=0))
# databig_mart.boxplot(column='Item_Visibility')


# # In[55]:


# databig_mart.hist(column='Item_Visibility')
# # From the box plot and histogram it is clearly understood that Item_Visibility is heavily right skewed and
# #there are so many outliers for this columns.One reason could be the minimum item visibility is zero,which is not appropriate


# # In[56]:


# # Univariate Analysis for Item_MRP
# print(databig_mart['Item_MRP'].describe())
# print(databig_mart['Item_MRP'].median())
# print(databig_mart['Item_MRP'].skew(axis=0))
# print(databig_mart['Item_MRP'].mode())
# databig_mart.boxplot(column='Item_MRP')


# # In[57]:


# databig_mart.hist(column='Item_MRP')
# #From the histogram we could see that Items of Mrp in range of 100 to 125 are having more purchases and after that from 175 to 190 are 
# #the second highest.We will have a look at the product which is in the range of 100 to 125 and 175 to 190


# # In[71]:


# databig_mart['Item_MRP'].values


# # In[18]:


# #Univariate Analysis for categorical variable
# categorical_columns = [x for x in databig_mart.dtypes.index if databig_mart.dtypes[x]=='object']
# #Exclude ID cols and source:
# categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
# for col in categorical_columns:
#     print ('\nFrequency of Categories for varible %s'%col)
#     print (databig_mart[col].value_counts())


# In[ ]:


#Item Fat content :Now from the categorical variables frequencies we will draw inferences
#Item Fat Content: Low fat content food is being preferred by most of the customers.As part of data correction 
#we need to convert all Low fat values to a single name such as Low Fat.
#Item Outlet Size: Outlets with Medium size are more being preferred and the sales outcome is more from these stores.
#Outlet_Location_type: Tier3 outlets are more being preferred and the sales outcome is more from these stores.
#Outlet_Type :Type 1 Supermarkets will be having more demand and the sales would be preferably more from these stores.


# In[4]:


# Bivariate Analysis on the data set
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,8))
sns.barplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',estimator=sum,data=databig_mart)


# In[20]:


plt.figure(figsize=(15,8))
sns.barplot(x='Outlet_Size',y='Item_Outlet_Sales',estimator=sum,data=databig_mart)


# In[23]:


plt.figure(figsize=(15,8))
sns.barplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',estimator=sum,data=databig_mart)


# In[34]:


plt.figure(figsize=(15,8))
sns.countplot('Item_Type',data=databig_mart,palette='spring')


# In[8]:


plt.figure(figsize=(15,8))
sns.barplot(x='Item_Outlet_Sales',y='Item_Type',estimator=sum,data=databig_mart)


# In[ ]:


#From both of the above graphs, we could see that fruits and vegetables has highest number of sales.


# In[6]:


# plt.figure(figsize=(15,8))
# plt.scatter(y='Item_Outlet_Sales',x='Item_MRP',data=train)
# plt.xlabel('Item MRP')
# plt.ylabel('Item Outlet Sales')


# In[ ]:


#From the above scatter plot we could see that Item MRP between 200 and 250 are having more Item outlet sales


# In[31]:


#Heat Map
#correlation Matrix need to be plotted for continous variables 
# databig_mart.columns
# databig_mart_corr=databig_mart[['Item_Weight','Item_Visibility','Item_Outlet_Sales']]
# sns.heatmap(databig_mart.corr(),annot=True,fmt=".2f") 
# plt.figure(figsize=(200,8))


# In[ ]:


#From the correlation matrix we could see that Item MRP is having good correlation of value 0.57 with Item outletsales.Lets keep this in mind
#and let us see whether our model also predicts the same


# In[7]:


#Multivariate Analysis
# plt.figure(figsize=(25,5))
# sns.barplot('Item_Type','Item_Outlet_Sales',hue='Item_Fat_Content',data=databig_mart,palette='mako')
# plt.legend()


# In[ ]:


#Now we could observe that fat content LF,low fat and Low Fat are come under Low Fat same applies reg.We need to correct this in the data set


# In[9]:


# plt.figure(figsize=(10,5))
# sns.barplot('Outlet_Location_Type','Item_Outlet_Sales',hue='Outlet_Type',data=databig_mart,palette='magma')
# plt.legend()


# In[11]:


# Missing values imputation
#Now for the missing values in both Item_Weight and Outlet_size there are no outliers ,we will impute Item_weight with mean of the Item_weight column 
#and outlet_size by mode of the column


# In[5]:


databig_mart['Item_Weight']= databig_mart['Item_Weight'].fillna(databig_mart['Item_Weight'].mean())

databig_mart['Item_Weight'].isnull().sum()


# In[6]:


databig_mart['Outlet_Size']=databig_mart['Outlet_Size'].fillna(databig_mart['Outlet_Size'].mode()[0])

databig_mart['Outlet_Size'].isnull().sum()


# In[ ]:


#Feature Engineering
#As part of Feature Engineering first we will modify the Item_Visibility column where rows are having as zero we will replace with the mean


# In[7]:


databig_mart['Item_Visibility']=databig_mart['Item_Visibility'].replace(0,databig_mart['Item_Visibility'].mean())

print(databig_mart['Item_Visibility'])

databig_mart['Item_Visibility'].describe()


# In[ ]:


#Creating a combined column for Identifier as Identifier name starts with FD,DR and NC .They are Food ,drinks and Non consumable


# In[8]:


databig_mart['Item_Combined']=databig_mart['Item_Identifier'].apply(lambda x:x[0:2])
databig_mart['Item_Combined'] = databig_mart['Item_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
databig_mart['Item_Combined'].value_counts()


# In[ ]:


#Now in the data we have output establishment Year ,to understand better and from the problem statement the data is been collected in the year 2013.Now as we are analyzingin this year 
 #create a new column such as No.of.Years of Establishment


# In[9]:


databig_mart['Outlet_Establishment_Year']=2020-databig_mart['Outlet_Establishment_Year']
databig_mart['Outlet_Establishment_Year'].describe()
databig_mart['Outlet_Establishment_Year'].head()


# In[32]:


plt.figure(figsize=(15,8))
sns.barplot(x='Outlet_Establishment_Year',y='Item_Outlet_Sales',estimator=sum,data=databig_mart)


# In[ ]:


#Now we will modify the Item_Fat_Content column 


# In[10]:


print ('\nOriginal Categories:')
print (databig_mart['Item_Fat_Content'].value_counts())

print ('\nModified Categories:')
databig_mart['Item_Fat_Content'] = databig_mart['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print (databig_mart['Item_Fat_Content'].value_counts())


# In[39]:


databig_mart.loc[databig_mart['Item_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
databig_mart['Item_Fat_Content'].value_counts()


# In[ ]:


#Encoding for categorical variables


# In[ ]:


#Label Encoding 


# In[11]:


from sklearn.preprocessing import LabelEncoder

l=LabelEncoder()
databig_mart['Outlet']=l.fit_transform(databig_mart['Outlet_Identifier'])
print(databig_mart['Outlet'])

var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Combined','Outlet_Type','Outlet']
l = LabelEncoder()
for i in var_mod:
    databig_mart[i] = l.fit_transform(databig_mart[i])
    

print(databig_mart.shape)


# In[ ]:


#In this process we could proceed with label encoding itself because of two reasons.
#1.All our categorical variables are ordinal 
#2.As this is a regression problem, when we convert into one hot encoding,there might be a problem of multi collinearity between the independent variables
#which violates the assumption of linear regression.


# In[ ]:


#Export our data


# In[ ]:


#Now we will drop the columns which we converted i.e Item_Identifier and Outlet_Identifier and will divide the data set into train
#and test data sets.


# In[]:
databig_mart.drop(['Item_Type','Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
print(databig_mart)


# In[ ]:


#Divide into test and train:
train = databig_mart.loc[databig_mart['source']=="train"]
test = databig_mart.loc[databig_mart['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)


# In[78]:


train.head()


# In[79]:


test.head()


# In[ ]:


#Model Building


# In[16]:


#Linear Regression
from sklearn.model_selection import cross_val_score

from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
Predictors=train[['Item_Weight','Item_Fat_Content','Item_Visibility','Item_Combined','Item_MRP','Outlet_Size','Outlet_Location_Type','Outlet_Type','Outlet_Establishment_Year','Outlet']]
Target=train[['Item_Outlet_Sales']]
Predictors_train,Predictors_test,Target_train,Target_test = train_test_split(Predictors,Target,test_size=0.2,random_state=22)
print(Predictors_train)
#print(Predictors_test)
#print(Target_train)
#print(Target_test)
f= ['Item_Weight','Item_Fat_Content','Item_Visibility','Item_Combined','Item_MRP','Outlet_Size','Outlet_Location_Type','Outlet_Type','Outlet_Establishment_Year','Outlet']
print(f)


# In[ ]:


#Linear Regression Model and Predictions on the train data set
LR = LinearRegression(normalize=True)
LR.fit(Predictors_train,Target_train)
predictions = LR.predict(Predictors_test)
print(predictions)


# In[88]:


import numpy as np
#Print model report:
print ("\nModel Report")
print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(Target_test,predictions)))
print ("MSE : %.4g"% (metrics.mean_squared_error(Target_test,predictions)))


# In[110]:


# Random Forest
from sklearn.ensemble import RandomForestRegressor
RF= RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4,random_state=22)

# Fitting the model on our trained dataset.
RF.fit(Predictors_train,Target_train)

# Making Predictions
y_pred = RF.predict(Predictors_test)


# In[111]:


coef3 = pd.Series(RF.feature_importances_, f).sort_values(ascending=False)
coef3


# In[121]:


#Perform cross-validation:
# Linear Regression
scores = cross_val_score(LinearRegression(), Predictors, Target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    


# In[122]:


#Cross Validation for Random Forest
scores = cross_val_score(RandomForestRegressor(), Predictors, Target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[1]:

import pickle

# Saving model to disk
pickle.dump(LR,open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[12.600000, 2, 0.041727,1,122.0072,1,2,2,21,3]]))


