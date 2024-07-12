# 1. Downloading the Stock Data
# We'll download the Tesla stock data from Yahoo Finance for the specified period.

import yfinance as yf

# Fetching the Tesla stock data for the past 5 years
tesla_stock_data = yf.download('TSLA', start='2018-01-01', end='2023-01-01')

# We will consider the Adjusted Close price
tesla_close = tesla_stock_data['Adj Close']

# Checking the head of the dataset
tesla_close.head()


# 2. Checking Stationarity using ADF Test
# We define a function to perform the ADF test and print the results.

import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Function to run the ADF test
def adf_test(timeseries):
    print('Results of Dickey-Fuller Test:')
    df_test = adfuller(timeseries, autolag='AIC')
    df_output = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in df_test[4].items():
        df_output['Critical Value (%s)' % key] = value
    print(df_output)

# Running the tests
adf_test(tesla_close)


# 3. Differencing the Data and Rechecking Stationarity
# We apply differencing to make the time series stationary and recheck it using the ADF test.
import matplotlib.pyplot as plt

# Applying differencing
tesla_close_diff = tesla_close.diff().dropna()

# Perform the ADF test again
adf_test_diff = adfuller(tesla_close_diff)

print('ADF Statistic after differencing: %f' % adf_test_diff[0])
print('p-value after differencing: %f' % adf_test_diff[1])
print('Critical Values after differencing:')
for key, value in adf_test_diff[4].items():
    print('\t%s: %.3f' % (key, value))

# Check for stationarity after differencing
if adf_test_diff[1] > 0.05:
    print("The time series is still not stationary after differencing.")
else:
    print("The time series is stationary after differencing.")

# 4. Plotting the Differenced Data
# We plot the differenced time series to visualize it.
# Plotting the differenced time series
plt.figure(figsize=(12,6))
plt.plot(tesla_close_diff, label='Differenced TSLA Stock Price')
plt.title('TSLA Stock Price after Differencing')
plt.xlabel('Date')
plt.ylabel('Differenced Price')
plt.legend()
plt.show()


# 5. Decomposing the Time Series
# We decompose the time series into trend, seasonality, and residuals.
from statsmodels.tsa.seasonal import seasonal_decompose

# Ensuring the time series has a frequency
tesla_close = tesla_close.asfreq('B')

# Interpolating missing values
tesla_close_interpolated = tesla_close.interpolate(method='linear')

# Decomposition of the time series
decomposition = seasonal_decompose(tesla_close_interpolated, model='multiplicative')

# Extracting the trend, seasonality, and residuals
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plotting the decomposition
plt.figure(figsize=(12,8))

plt.subplot(411)
plt.plot(tesla_close, label='Original')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()




# 6. Decomposing the Differenced Time Series
# Interpolating missing values for the differenced series
tesla_close_diff_interpolated = tesla_close_diff.asfreq('B').interpolate(method='linear')

# Decomposition of the differenced time series
decomposition_diff = seasonal_decompose(tesla_close_diff_interpolated, model='additive')

# Extracting the trend, seasonality, and residuals
trend_diff = decomposition_diff.trend
seasonal_diff = decomposition_diff.seasonal
residual_diff = decomposition_diff.resid

# Plotting the decomposition of the differenced series
plt.figure(figsize=(12,8))

plt.subplot(411)
plt.plot(tesla_close_diff_interpolated, label='Differenced')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(trend_diff, label='Trend')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(seasonal_diff,label='Seasonality')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(residual_diff, label='Residuals')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()