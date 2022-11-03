#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_boston


# In[3]:


load_boston()


# In[ ]:





# In[4]:


boston = load_boston()
#pd.DataFrame(boston.data).head()


# In[5]:


boston


# In[6]:


print(boston.keys())


# In[7]:


print(boston.DESCR)


# In[8]:


print(boston.data)  # Independent feature 


# In[9]:


print(boston.target)


# In[10]:


print(boston.feature_names)


# In[11]:


print(boston.feature_names)


# In[12]:


# Let's Preparing the Dataframe 
dataset=pd.DataFrame(boston.data,columns=boston.feature_names)


# In[13]:


dataset.head()


# In[14]:


dataset['price']=boston.target


# In[15]:


dataset.head()


# In[16]:


dataset.info()


# In[17]:


dataset.describe()


# In[18]:


# Correlation with the feature. We can also use heatmap 

dataset.corr()


# In[19]:


sns.pairplot(dataset)


# In[20]:


sns.set(rc={'figure.figsize':(8,6)})
sns.heatmap(dataset.corr(),annot=True)


# In[21]:


plt.scatter(dataset['CRIM'],dataset['price'])
plt.xlabel("Crime Rate")
plt.ylabel("Price")


# In[22]:


plt.scatter(dataset['RM'],dataset['price'])
plt.xlabel("RM")
plt.ylabel("Price")


# In[23]:


# If we have one Independent feature like RM and one dependent feature i.e. Price and use regplot to generate best fit line
# with Ridge and Lasso.

sns.regplot(x="RM",y='price',data=dataset)


# In[24]:


sns.regplot(x="LSTAT",y='price',data=dataset)


# In[25]:


# For checking outlier we use BoxPlot
sns.boxplot(dataset['CRIM'])


# In[26]:


# Training the Model 
# Independen and Dependent Feature 
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1:]
#df.iloc[:, 0:2]


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


# In[29]:


X_train.shape


# In[30]:


X


# In[31]:


y_train.shape


# In[32]:


X_train


# In[33]:


y_test.shape


# In[34]:


# Feature Scaling or stadarization the dataset .In standarization Mean will be 0 and standard deviaion will be 1 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[ ]:





# In[35]:


X_train=scaler.fit_transform(X_train) # Never use standarization in o/p feature 


# In[36]:


X_train


# In[37]:


X_test=scaler.transform(X_test)    # To avoid data leakage 


# In[38]:


X_test


# In[39]:


# Now Model Training. It is Multiple Liner regression problem 
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression


# In[40]:


regression.fit(X_train,y_train)


# In[ ]:





# In[41]:


# Print the Coefficient and intersecept
print(regression.coef_)


# In[42]:


print(regression.intercept_)


# In[43]:


# Pridiction for the test data 
reg_pred=regression.predict(X_test)


# In[44]:


reg_pred  # This is predicted Data and y_test is truth point


# In[45]:


# Assumption of Liner regression 
plt.scatter(y_test,reg_pred)
plt.xlabel("Truh Data")
plt.ylabel("Pridited Data")


# In[46]:


# Residulas:- Truth - pidicted
residulas=y_test-reg_pred


# In[47]:


residulas


# In[48]:


sns.displot(residulas,kind='kde')


# In[49]:


# Scatter plot with pridiction and residuals
# Uniform Distribution:- Means no shape 
plt.scatter(reg_pred,residulas)


# In[50]:


# Perfromace Matrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[51]:


print(mean_squared_error(y_test,reg_pred))
print(mean_absolute_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# In[52]:


# R2 and adjusted R2
from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)


# In[53]:


from sklearn.linear_model import Ridge
ridge=Ridge()
ridge


# In[54]:


ridge.fit(X_train,y_train)


# In[55]:


print(ridge.coef_)


# In[56]:


ridge_pred=ridge.predict(X_test)


# In[76]:


print(ridge_pred)


# In[57]:


# Assumption of Ridge regression 
plt.scatter(y_test,ridge_pred)
plt.xlabel("Truh Data")
plt.ylabel("Pridited Ridge Data")


# In[58]:


# Residulas for Ridge :- Truth - pidicted
residulas_ridge=y_test-ridge_pred


# In[59]:


sns.displot(residulas_ridge,kind='kde')


# In[60]:


# This is for Ridge 
# Scatter plot with pridiction and residuals
# Uniform Distribution:- Means no shape 
plt.scatter(ridge_pred,residulas_ridge)


# In[61]:


# Performace Matrics
print(mean_squared_error(y_test,ridge_pred))
print(mean_absolute_error(y_test,ridge_pred))
print(np.sqrt(mean_squared_error(y_test,ridge_pred)))


# In[98]:


# R2 and Adjusted R2
score_ridge=r2_score(y_test,ridge_pred)
print(score_ridge) 
#print(score)


# In[65]:


# Implementation for Lasso 
from sklearn.linear_model import Lasso
lasso=Lasso()
lasso


# In[66]:


lasso.fit(X_train,y_train)


# In[70]:


lasso_pred=lasso.predict(X_test)


# In[71]:


print(lasso_pred)


# In[89]:


# Assumption of Lasso regression 
plt.scatter(y_test,lasso_pred)
plt.xlabel("Truh Data")
plt.ylabel("Pridited Lasso Data")


# In[99]:


# Residulas for Lasso :- Truth - pidicted
y_test-lasso_pred
#sns.displot(residulas_ridge,kind='kde')


# In[79]:


# Performace Matrics
print(mean_squared_error(y_test,lasso_pred))
print(mean_absolute_error(y_test,lasso_pred))
print(np.sqrt(mean_squared_error(y_test,lasso_pred)))


# In[94]:


# R2 and Adjusted r2
score_lasso=r2_score(y_test,lasso_pred)
print(score_lasso) 


# In[82]:


# ElasticNet
from sklearn.linear_model import ElasticNet
elasticNet=ElasticNet()
elasticNet


# In[83]:


elasticNet.fit(X_train,y_train)


# In[86]:


elec_pred=elasticNet.predict(X_test)


# In[88]:


# Assumption of Ridge regression 
plt.scatter(y_test,elec_pred)
plt.xlabel("Truh Data")
plt.ylabel("Pridited elasticNet Data")


# In[91]:


# Residulas for elastic net :- Truth - pidicted
residulas_elec=y_test-elec_pred


# In[92]:


print(elec_pred)


# In[93]:


# Performace Matrics
print(mean_squared_error(y_test,elec_pred))
print(mean_absolute_error(y_test,elec_pred))
print(np.sqrt(mean_squared_error(y_test,elec_pred)))


# In[95]:


# R2 and Adjusted r2
score_elec=r2_score(y_test,elec_pred)
print(score_elec) 


# In[ ]:




