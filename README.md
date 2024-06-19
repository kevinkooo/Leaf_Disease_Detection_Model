# Leaf Disease Detection Model
## Objective:



## Background
Plant health is crucial for argricultural productivity. Traditional methods of desease detection often rely on manual inspection, which can be time_comsuming and vulnerable to human error. Each year, plan diseases- ranging from coffee leaf rust to banana fusarium wilt - cost the global economy more than $220 billion. In fact, 40% of global crop production is lost to pests. Thus, with advancements in deep learning and computer vision techniques, there's a growing opportunity to automate and improve the accuracy of plant disease recognition.

## Data
This project uses dataset from Kaggle. The dataset contains images on three groups of leaves - healthy, powdery, and rust. There are a total of 1530 images divided into train, test and validation sets. The splits are as follows:
- 458 healhty
- 430 powder
- 434 rust

  
The test set contains 50 images of each group and the validation set contains 20 of each group.


## Model- Convolutional Neural Network (CNN)


## Testing and validation


## Improvement and Recommendation





#!/usr/bin/env python
# coding: utf-8

# # Assignment 1
# ### Kevin Ko

# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import kpss, adfuller
from pandas.plotting import autocorrelation_plot


# In[2]:


df = pd.read_csv(r"C:\Users\kevin.ko\Downloads\hw1_data.csv")
df.head()


# ## A.	10 pts Task: Perform EDA on the dataset. Include both quantitative and qualitative descriptions. Check for missing data and correlations.

# In[3]:


# Data Overview
df.info()

df.describe()


# In[4]:


# checking for null
df.isnull().sum()


# In[5]:


# checking for correlation
correlation_matrix = df[['IWM', 'QQQ', 'SPY']].corr()
correlation_matrix


# In[6]:


plt.figure(figsize=(10, 6))

# Plotting IWM
plt.plot(df['Date'], df['IWM'], label='IWM', marker='o')
# Plotting QQQ
plt.plot(df['Date'], df['QQQ'], label='QQQ', marker='o')
# Plotting SPY
plt.plot(df['Date'], df['SPY'], label='SPY', marker='o')

# Adding title and labels
plt.title('ETF Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()


# ## B.	5 pts Concept: Describe the data. Do you expect this data to be stationary? Do you expect this data to be a random walk?

# This data is not stationary. Even though we can see that it does not have a huge variance throughout the period and no huge seasonal impact, there is a clear long term increase in the level of data, indicating a upward trend. The data may follow a random walk pattern because stock prices can be impacted by exterior factors heavily and may not always be predictable based on past data.

# ## C.	10 pts Task: Create a function that returns the stationarity test results from both ADF and KPSS tests

# In[11]:


def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

alpha = 0.05

#define KPSS
def kpss_test(timeseries, trend='c'):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression=trend)
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)


# ## D.	5 pts Task: Use the function to test if each series is stationary. Return results from both ADF and KPSS tests for all three time series

# ### IWM

# In[12]:


# adf test
adf_test(df['IWM'])

ad_fuller_results = adfuller(df['IWM'])
ad_fuller_pval = ad_fuller_results[1]
ad_fuller_bool = ad_fuller_pval <= alpha

print(f'stationarity from ad_fuller test: {ad_fuller_bool}')


# In[13]:


#kpss test
kpss_test(df['IWM'])


# In[15]:


kpss_test(df['IWM'], trend='ct')


# In[17]:


kpss_test_results = kpss(df['IWM'])
kpss_pval = kpss_test_results[1]
kpss_test_bool = kpss_pval >= alpha

print(f'stationarity from KPSS test: {kpss_test_bool}')


# ### QQQ

# In[18]:


# adf test
adf_test(df['QQQ'])

ad_fuller_results = adfuller(df['QQQ'])
ad_fuller_pval = ad_fuller_results[1]
ad_fuller_bool = ad_fuller_pval <= alpha

print(f'stationarity from ad_fuller test: {ad_fuller_bool}')


# In[19]:


#kpss test
kpss_test(df['QQQ'])


# In[20]:


kpss_test(df['QQQ'], trend='ct')


# In[21]:


kpss_test_results = kpss(df['QQQ'])
kpss_pval = kpss_test_results[1]
kpss_test_bool = kpss_pval >= alpha

print(f'stationarity from KPSS test: {kpss_test_bool}')


# ### SPY

# In[22]:


# adf test
adf_test(df['SPY'])

ad_fuller_results = adfuller(df['SPY'])
ad_fuller_pval = ad_fuller_results[1]
ad_fuller_bool = ad_fuller_pval <= alpha

print(f'stationarity from ad_fuller test: {ad_fuller_bool}')


# In[23]:


#kpss test
kpss_test(df['SPY'])


# In[24]:


kpss_test(df['SPY'], trend='ct')


# In[25]:


kpss_test_results = kpss(df['SPY'])
kpss_pval = kpss_test_results[1]
kpss_test_bool = kpss_pval >= alpha

print(f'stationarity from KPSS test: {kpss_test_bool}')


# Case 1: Both tests conclude that the given series is stationary – The series is stationary
# Case 2: Both tests conclude that the given series is non-stationary – The series is non-stationary
# Case 3: ADF concludes non-stationary, and KPSS concludes stationary – The series is trend stationary. To make the series strictly stationary, the trend needs to be removed in this case. Then the detrended series is checked for stationarity.
# Case 4: ADF concludes stationary, and KPSS concludes non-stationary – The series is difference stationary. Differencing is to be used to make series stationary. Then the differenced series is checked for stationarity.

# ## E.	5 pts Task: Difference the three time series and return stationarity results from both ADF and KPSS tests

# In[26]:


differenced_IWM_data = (df['IWM']-df['IWM'].shift()).iloc[1:]
differenced_QQQ_data = (df['QQQ']-df['QQQ'].shift()).iloc[1:]
differenced_SPY_data = (df['SPY']-df['SPY'].shift()).iloc[1:]


# In[27]:


# ADF Test
ad_fuller_results = adfuller(differenced_IWM_data)
ad_fuller_pval = ad_fuller_results[1]
ad_fuller_bool = ad_fuller_pval <= alpha
print(f'stationarity from ad_fuller test for IWM: {ad_fuller_bool}')

ad_fuller_results = adfuller(differenced_QQQ_data)
ad_fuller_pval = ad_fuller_results[1]
ad_fuller_bool = ad_fuller_pval <= alpha
print(f'stationarity from ad_fuller test for QQQ: {ad_fuller_bool}')


ad_fuller_results = adfuller(differenced_SPY_data)
ad_fuller_pval = ad_fuller_results[1]
ad_fuller_bool = ad_fuller_pval <= alpha
print(f'stationarity from ad_fuller test for SPY: {ad_fuller_bool}')


# In[28]:


# KPSS Test
kpss_test_results = kpss(differenced_IWM_data)
kpss_pval = kpss_test_results[1]
kpss_test_bool = kpss_pval >= alpha

print(f'stationarity from KPSS test IWM: {kpss_test_bool}')

kpss_test_results = kpss(differenced_QQQ_data)
kpss_pval = kpss_test_results[1]
kpss_test_bool = kpss_pval >= alpha

print(f'stationarity from KPSS test QQQ: {kpss_test_bool}')

kpss_test_results = kpss(differenced_SPY_data)
kpss_pval = kpss_test_results[1]
kpss_test_bool = kpss_pval >= alpha

print(f'stationarity from KPSS test SPY: {kpss_test_bool}')


# ## F.	10 pts Concept: In general, explain how you can get different stationarity results from the two tests. 

# ## G.	5 pts Concept: What does trend stationary mean?

# A trend stationary series fluctuates around a deterministic trend (the mean of the series) with no tendency for the amplitude of the fluctuations to increase or decrease.
# 

# # 2. Random Walk

# ### A.	5 pts Task: Using random.sample create a sample with length 1000 of 1 if >=0.5 and -1 if less than 0.5

# In[32]:


import random
import numpy as np


# In[31]:


sample = [1 if random.random() >= 0.5 else -1 for _ in range(1000)]
sample[:10]  # Display the first 10 elements for verification


# ### B.	5 pts Task: Use cumulative sum to create movement

# In[33]:


movement = np.cumsum(sample)
movement[:10]  # Display the first 10 elements for verification


# ### C.	5 pts Concept: Describe which of the two series (2a or 2b) is a random walk, why? What is the other series?

# 2b is a random walk because it represents the cumulative sum of the random steps generated in part 2a.

# ### D.	5 pts Task: Create two plots of the random walk time series, use .plot() and use autocorrelation_plot()

# In[35]:


plt.figure(figsize=(10, 6))
plt.plot(movement)
plt.title("Random Walk Time Series")
plt.xlabel("Time")
plt.ylabel("Position")
plt.show()

# Plot the autocorrelation
plt.figure(figsize=(10, 6))
autocorrelation_plot(movement)
plt.title("Autocorrelation Plot of Random Walk")
plt.show()


# ### E.	10 pts Concept: Is the random walk stationary? Defend using quantitative tests.

# In[36]:


adf_result = adfuller(movement)
adf_result


# ### F.	5 pts Task: Difference the data and show the autocorrelation plot

# In[41]:


differenced_movement = np.diff(movement)

# Plot the autocorrelation of the differenced data
plt.figure(figsize=(10, 6))
autocorrelation_plot(differenced_movement)
plt.ylim(-0.2, 0.2)
plt.title("Autocorrelation Plot of Differenced Random Walk")
plt.show()


# ### G.	10 pts Concept: What is the significance of this correlation plot?

# The autocorrelation plot of the differenced data should show little to no significant autocorrelation, indicating that differencing has removed any trends or patterns in the original series. This implies that the differenced series is stationary, which is a key property for many time series analysis methods.

# ### H.	5 pts Concept: Can a random walk be predicted? Why or why not?

# A random walk is generally not predictable because each step is independent of the previous steps and is determined by a random process. The future value of the series is only influenced by the most recent value and a random shock, making it inherently unpredictable.

# In[ ]:





