#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight') # For plots


# In[4]:


pjme = pd.read_csv('/Users/amberowens/Desktop/hourly-energy-consumption/PJME_hourly.csv',
                   index_col=[0], parse_dates=[0])


# In[5]:


# Color pallete for plotting
color_pal = ["#F8766D", "#D39200", "#93AA00",
             "#00BA38", "#00C19F", "#00B9E3",
             "#619CFF", "#DB72FB"]
pjme.plot(style='.', figsize=(15,5), color=color_pal[0], title='PJM East')
plt.show()


# In[6]:


def create_features(df, label=None):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X

X, y = create_features(pjme, label='PJME_MW')

features_and_target = pd.concat([X, y], axis=1)
# See our features and target
features_and_target.tail()


# In[ ]:





# In[9]:


features_and_target


# In[8]:


split_date = '01-Jan-2015'
pjme_train = pjme.loc[pjme.index <= split_date].copy()
pjme_test = pjme.loc[pjme.index > split_date].copy()


# In[9]:


# Plot train and test so you can see where we have split
pjme_test     .rename(columns={'PJME_MW': 'TEST SET'})     .join(pjme_train.rename(columns={'PJME_MW': 'TRAINING SET'}),
          how='outer') \
    .plot(figsize=(15,5), title='PJM East', style='.')
plt.show()


# In[10]:


pjme_train.reset_index()     .rename(columns={'Datetime':'ds',
                     'PJME_MW':'y'}).head()


# In[11]:


# Setup and train model and fit
model = Prophet()
model.fit(pjme_train.reset_index()               .rename(columns={'Datetime':'ds',
                               'PJME_MW':'y'}))


# In[12]:


# Predict on training set with model
pjme_test_fcst = model.predict(df=pjme_test.reset_index()                                    .rename(columns={'Datetime':'ds'}))


# In[14]:


pjme_test_fcst.tail()


# In[15]:


# Plot the forecast
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
fig = model.plot(pjme_test_fcst,
                 ax=ax)
plt.show()


# In[16]:


# Plot the components of the model
fig = model.plot_components(pjme_test_fcst)


# In[17]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(pjme_test.index, pjme_test['PJME_MW'], color='r')
fig = model.plot(pjme_test_fcst, ax=ax)


# In[40]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(pjme_test.index, pjme_test['PJME_MW'], color='r')
fig = model.plot(pjme_test_fcst, ax=ax)
ax.set_xbound(lower='01-01-2018',
              upper='08-01-2018')
ax.set_ylim(0, 60000)
plot = plt.suptitle(' 2018 Forecast vs Actuals')


# In[35]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(pjme_test.index, pjme_test['PJME_MW'], color='r')
fig = model.plot(pjme_test_fcst, ax=ax)
ax.set_xbound(lower='07-01-2018', upper='08-08-2018')
ax.set_ylim(0, 60000)
plot = plt.suptitle('Last  month of 2018 Forecast vs Actuals')


# In[37]:


mean_squared_error(y_true=pjme_test['PJME_MW'],
                   y_pred=pjme_test_fcst['yhat'])


# In[38]:


mean_absolute_error(y_true=pjme_test['PJME_MW'],
                   y_pred=pjme_test_fcst['yhat'])


# In[39]:


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(y_true=pjme_test['PJME_MW'],
                   y_pred=pjme_test_fcst['yhat'])


# In[ ]:




