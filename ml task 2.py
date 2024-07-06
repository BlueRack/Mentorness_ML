#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
# Load the dataset
file_path = 'household_power_consumption.txt'  
df = pd.read_csv(file_path, sep=';', parse_dates={'Datetime': ['Date', 'Time']}, infer_datetime_format=True)


# In[9]:


# Display basic information about the dataset
print(df.info())


# In[10]:


# Data preprocessing
# Handle missing values if any
print("Missing values before handling:")
print(df.isnull().sum())


# In[11]:


df.dropna(inplace=True)


# In[12]:


# Handle missing values after conversion
print("Missing values after handling:")
print(df.isnull().sum())


# In[13]:


# Combine date and time columns into a single datetime column
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H:%M:%S')

# Set Datetime column as index
df.set_index('Datetime', inplace=True)


# In[15]:


df.index = pd.to_datetime(df.index)


# In[17]:


df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')


# In[18]:


# Visualize global active power over time
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Global_active_power'], label='Global Active Power', color='blue')
plt.title('Global Active Power Over Time')
plt.xlabel('Time')
plt.ylabel('Global Active Power')
plt.grid(True)
plt.legend()
plt.show()


# In[19]:


decomposition = seasonal_decompose(df['Global_active_power'], model='additive', period=30) 
plt.figure(figsize=(14, 8))
decomposition.plot()
plt.suptitle('Seasonal Decomposition of Global Active Power')
plt.show()


# In[20]:


# Splitting data into train and test sets
train_size = int(len(df) * 0.8)  # 80% train, 20% test
train, test = df[:train_size], df[train_size:]


# In[25]:


# Fit ARIMA model on downscaled data
model_arima = ARIMA(train['Global_active_power'], order=(1,1,0))
model_arima_fit = model_arima.fit()



# In[26]:


# Forecasting with ARIMA
forecast_arima = model_arima_fit.forecast(steps=len(test))


# In[27]:


# Model evaluation
rmse_arima = np.sqrt(mean_squared_error(test['Global_active_power'], forecast_arima))
print(f"ARIMA RMSE: {rmse_arima}")


# In[29]:


# Visualize predictions
plt.figure(figsize=(14, 7))
plt.plot(test.index, test['Global_active_power'], label='Actual')
plt.plot(test.index, forecast_arima, label='ARIMA Forecast', linestyle='--')
plt.title('ARIMA  Forecasts vs Actual')
plt.xlabel('Time')
plt.ylabel('Global Active Power')
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:




