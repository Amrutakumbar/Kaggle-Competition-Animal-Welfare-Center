#!/usr/bin/env python
# coding: utf-8

# # IMPORT THE LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # LOAD THE DATASET

# In[2]:


train = pd.read_csv("train.csv",encoding="ISO-8859-1")
train


# In[3]:


test = pd.read_csv("test.csv",encoding="ISO-8859-1")
test


# # DATA PREPROCESSING

# In[4]:


train.info()


# ## Relation between columns and their visualisation

# In[5]:


import seaborn as sns
sns.countplot(x='animal_id_outcome', data=train)


# In[6]:


import seaborn as sns
sns.countplot(x='animal_type', data=train)


# ## Checking for Null values

# In[7]:


train.isna().sum()

import missingno as msno
msno.bar(train)
# In[8]:


#% of null values
data_percentage=print('percentage of missing values'.upper())
train.isna().sum()/train.shape[0]*100


# In[9]:


cateogry_columns=train.select_dtypes(include=['category']).columns.tolist()
integer_columns=train.select_dtypes(include=['bool']).columns.tolist()

for column in train:
    if train[column].isnull().any():
        if(column in cateogry_columns):
            train[column]=train[column].fillna(train[column].mode()[0])
        else:
            train[column]=train[column].fillna(train[column].mean)


# In[10]:


train.isna().sum()


# + NO NULL VALUES

# ### Drop the unneccesory columns

# In[11]:


train.columns


# In[12]:


drop=["outcome_datetime","intake_datetime",
                    'date_of_birth','intake_monthyear','intake_datetime',
                    'outcome_monthyear','outcome_datetime','time_in_shelter',
                   'age_upon_intake','age_upon_outcome','age_upon_intake_(days)','age_upon_intake_(years)','age_upon_outcome_(days)',
                'age_upon_outcome_(years)','intake_number','sex_upon_intake','sex_upon_outcome']
train.drop(drop,axis=1,inplace=True)


# # LABEL ENCODING

# In[13]:


train.info()


# In[14]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
train['dob_year']=label.fit_transform(train.dob_year)
train['dob_month']=label.fit_transform(train.dob_month)
train['animal_type']=label.fit_transform(train.animal_type)
train['breed']=label.fit_transform(train.breed)
train['color']=label.fit_transform(train.color)
train['intake_condition']=label.fit_transform(train.intake_condition)
train['intake_type']=label.fit_transform(train.intake_type)
#train['sex_upon_intake']=label.fit_transform(train.sex_upon_intake)
#train['count']=label.fit_transform(train.count)
train['age_upon_intake_age_group']=label.fit_transform(train.age_upon_intake_age_group)
train['intake_month']=label.fit_transform(train.intake_month)
train['intake_year']=label.fit_transform(train.intake_year)
train['intake_weekday']=label.fit_transform(train.intake_weekday)
train['intake_hour']=label.fit_transform(train.intake_hour)
train['time_in_shelter_days']=label.fit_transform(train.time_in_shelter_days)
#train['sex_upon_outcome']=label.fit_transform(train.sex_upon_outcome)
train['age_upon_outcome_age_group']=label.fit_transform(train.age_upon_outcome_age_group)
train['outcome_month']=label.fit_transform(train.outcome_month)
train['outcome_year']=label.fit_transform(train.outcome_year)
train['outcome_weekday']=label.fit_transform(train.outcome_weekday)
train['outcome_hour']=label.fit_transform(train.outcome_hour)
train['outcome_number']=label.fit_transform(train.outcome_number)
train['outcome_type']=label.fit_transform(train.outcome_type)
train['animal_id_outcome']=label.fit_transform(train.animal_id_outcome)


# In[15]:


train.head()


# In[16]:


train.info()


# # Splitting the train dataset

# In[17]:


X= train.iloc[:,train.columns!='outcome_type']
y =train.outcome_type


# In[18]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# In[19]:


print(X_train.shape), print(y_train.shape)
print(X_test.shape), print(y_test.shape)


# # Feature Selection: Using Chi square

# In[20]:


import sklearn
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

chi2 = SelectKBest(score_func=chi2, k='all')
model_chi2 = chi2.fit(X_train, y_train)

print(model_chi2.scores_)
features_select =model_chi2.transform(X_train)


# In[21]:


features=np.array([1.72917094e+03 ,5.38065355e+01, 1.13042704e+03 ,8.02876621e+04,
 3.42201076e+04, 3.31365597e+02, 4.06102114e+02, 6.15095637e-29,
 1.47209790e+04, 1.70172590e+02, 3.88128262e+02, 3.30027505e+01,
 5.15811991e+01, 4.88213324e+07, 1.46166502e+04, 1.80968589e+02,
 3.62789908e+02, 5.68014305e+01, 2.13787162e+03, 3.49009300e+03])


# In[22]:


features=pd.DataFrame(features)


# In[23]:


features.plot.bar()


# # Feature Selection:Using Mutual info classifier

# In[24]:


from sklearn.feature_selection import mutual_info_classif

mic = SelectKBest(score_func=mutual_info_classif, k='all')
model_mic=mic.fit(X_train, y_train)
features_select = model_mic.transform(X_train)


# In[25]:


for i in range(len(model_mic.scores_)):
    print('Feature %d: %f' % (i, model_mic.scores_[i]))
    
plt.bar([i for i in range(len(model_mic.scores_))], model_mic.scores_)
plt.show()  


# # Decision tree

# In[26]:


from sklearn.tree import  DecisionTreeClassifier
model_dt=DecisionTreeClassifier()
model_dt.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
results = cross_val_score(model_dt,X_train, y_train)
print(results.mean())


# In[27]:


#get importance
imp = model_dt.feature_importances_
#summarize feature importance
for i,v in enumerate(imp):
   print('Feature: %0d, Score: %.5f' % (i,v))
#plot feature importance
plt.bar([x for x in range(len(imp))], imp)
plt.show()


# # Final columns to be dropped are as follows

# In[28]:


new_data =train


# In[29]:


train = train.drop(columns=["age_upon_intake_age_group","age_upon_outcome_age_group",'count',
       'intake_year', 'intake_weekday','outcome_year', 'outcome_weekday' ])
train.shape


# In[30]:


train.columns


# ### Splitting the dataset

# In[31]:


from sklearn.model_selection import train_test_split
x_train,x_test,Y_train,Y_test = train_test_split(new_data,y,test_size=0.2, random_state=42)


# In[32]:


print(x_train.shape), print(Y_train.shape)
print(x_test.shape), print(Y_test.shape)


# In[33]:


new_data.info()


# # 1)LOGISTIC REGRESSION

# In[34]:


from sklearn.linear_model import LogisticRegression
model_log=LogisticRegression(solver='liblinear')
model_log.fit(x_train,Y_train)


# In[35]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Y_pred = model_log.predict(x_test)
print(accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


# In[37]:


from sklearn.metrics import f1_score
f1_score(Y_test, Y_pred, average='micro')


# # 2)Decision Tree

# In[38]:


from sklearn.tree import  DecisionTreeClassifier

model_dt=DecisionTreeClassifier()
model_dt.fit(x_train,Y_train)


# In[39]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Y_pred = model_dt.predict(x_test)
print(accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


# In[40]:


from sklearn.metrics import f1_score
f1_score(Y_test, Y_pred, average='micro')


# # 3)Random Forest

# In[41]:


from sklearn.ensemble import RandomForestClassifier

num_trees = 100
model_rf = RandomForestClassifier(n_estimators=num_trees)
model_rf.fit(x_train,Y_train)


# In[42]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Y_pred = model_rf.predict(x_test)
print(accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


# In[43]:


from sklearn.metrics import f1_score
f1_score(Y_test, Y_pred, average='micro')


# # 4)Support vector Classifier

# In[44]:


from sklearn.svm import SVC

model_svc= SVC()
model_svc.fit(x_train,Y_train)


# In[45]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Y_pred = model_svc.predict(x_test)
print(accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


# In[46]:


from sklearn.metrics import f1_score
f1_score(Y_test, Y_pred, average='micro')


# # 5)Gausian NB

# In[47]:


from sklearn.naive_bayes import GaussianNB

model_nb = GaussianNB()
model_nb.fit(x_train,Y_train)


# In[48]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Y_pred = model_nb.predict(x_test)
print(accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


# In[49]:


from sklearn.metrics import f1_score
f1_score(Y_test, Y_pred, average='micro')


# # So here decision tree is giving 100% accuracy,so ignoring this one and random forest giving best accuracy,so predicting the model on random forest.

# # TEST DATA SET

# In[50]:


drop=["outcome_datetime","intake_datetime",
                    'date_of_birth','intake_monthyear','intake_datetime',
                    'outcome_monthyear','outcome_datetime','time_in_shelter',
                   'age_upon_intake','age_upon_outcome','age_upon_intake_(days)','age_upon_intake_(years)','age_upon_outcome_(days)',
                'age_upon_outcome_(years)','intake_number','sex_upon_intake','sex_upon_outcome']
test.drop(drop,axis=1,inplace=True)


# In[51]:


test = test.drop(columns=["age_upon_intake_age_group","age_upon_outcome_age_group",'count',
       'intake_year', 'intake_weekday','outcome_year', 'outcome_weekday' ])
test.shape


# In[52]:


test.columns


# In[53]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
test['dob_year']=label.fit_transform(test.dob_year)
test['dob_month']=label.fit_transform(test.dob_month)
test['animal_type']=label.fit_transform(test.animal_type)
test['breed']=label.fit_transform(test.breed)
test['color']=label.fit_transform(test.color)
test['intake_condition']=label.fit_transform(test.intake_condition)
test['intake_type']=label.fit_transform(test.intake_type)
test['intake_month']=label.fit_transform(test.intake_month)
test['intake_hour']=label.fit_transform(test.intake_hour)
test['time_in_shelter_days']=label.fit_transform(test.time_in_shelter_days)
test['outcome_month']=label.fit_transform(test.outcome_month)
test['outcome_hour']=label.fit_transform(test.outcome_hour)
test['outcome_number']=label.fit_transform(test.outcome_number)
test['animal_id_outcome']=label.fit_transform(test.animal_id_outcome)


# In[54]:


test.head()


# In[55]:


X.shape,X_test.shape


# In[56]:


y.shape


# In[57]:


from sklearn.ensemble import RandomForestClassifier

y = train["outcome_type"]

features = ['dob_year', 'dob_month', 'animal_type', 'breed', 'color',
       'intake_condition', 'intake_type', 'intake_month', 'intake_hour',
       'time_in_shelter_days', 'outcome_month', 'outcome_hour',
       'outcome_number']
X= train.iloc[:,train.columns!='outcome_type']
X_test = test.iloc[:,test.columns!='outcome_type']

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)


# In[58]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
output = pd.read_csv("sample_submission.csv")

output["outcome_type"].value_counts() # Predicted Values
#pred_Values = df1["outcome_type"].value_counts(normalize=True) #For relative Frequencies
pred_Values = output["outcome_type"].value_counts()
plt.figure(figsize=(12,8))
sns.barplot(pred_Values.index, pred_Values.values, alpha=1.0, dodge=False)
plt.title('PREDICTED VALUES BY THE MODEL, ABOUT ANIMAL STATE ON 31,809 VALUES OF TEST DATA')
plt.ylabel('Number of Animals', fontsize=12)
plt.xlabel('Animal State', fontsize=12)
plt.show()


# In[59]:


output_train = pd.read_csv("train.csv")

output_train["outcome_type"].value_counts() #Original DataSet Target Values
#X2["outcome_type"].value_counts(dropna=False)
original_Values = output_train["outcome_type"].value_counts()
plt.figure(figsize=(12,8))
sns.barplot(original_Values.index, original_Values.values, alpha=1.0, dodge=False)
plt.title('VALUES GIVEN IN THE TRAIN DATASET , ABOUT ANIMAL STATE ON 47,809 VALUES OF TRAIN DATA')
plt.ylabel('Number of Animals', fontsize=12)
plt.xlabel('Animal State', fontsize=12)
plt.show()


# In[60]:


output = pd.DataFrame({'animal_id_outcome': test.animal_id_outcome, 'outcome_type': predictions})
output


# + From above visualisation outcome type 0=adoption,1=transfer,2=return to owner,3=Euthanasia,4=Died,5=Missing,6=Relocate,7=Rto-Adoptan, 8=Disposal are belongs

# + And I have converted 'animal_id_outcome' to the numberic by using label encoder for better model building

# + Train and test dataset is different as some of Euthanasia,missing,relocate,Rto-Adopt and desposal are imbalanced while predicting

# In[61]:


output.to_csv('challenge_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




