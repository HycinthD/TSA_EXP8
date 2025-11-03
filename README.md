# Exp.no: 8 IMPLEMENTATION OF EXPONENTIAL SMOOTHING MODEL  
# Date: 03/11/2025  

# AIM:  
To implement the Exponential Smoothing model for forecasting Amazon stock closing prices using Python.

# ALGORITHM:  
1. Import necessary libraries (NumPy, Pandas, Matplotlib, Statsmodels, Scikit-learn).  
2. Load the dataset and preprocess it.  
   - Convert ‘Date’ to datetime format and set as index.  
   - Extract the ‘Close’ column.  
3. Visualize the original time series data.  
4. Compute and plot moving averages (rolling means).  
5. Resample the dataset to monthly frequency.  
6. Normalize the data using Min-Max Scaling.  
7. Split the dataset into training (80%) and testing (20%) sets.  
8. Fit the Exponential Smoothing model with additive trend and multiplicative seasonality.  
9. Forecast future values and plot training, testing, and prediction results.  
10. Evaluate model performance using RMSE.  
11. Refit the model on the full dataset and forecast the next 12 months.

# PROGRAM:  
```python
#Name: Hycinth D
# Reg No:212223240055
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Read dataset
data = pd.read_csv("/content/Amazon.csv")

# Convert 'Date' to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Extract Close column
close_data = data[['Close']]
print("Shape of the dataset:", close_data.shape)
print("First 10 rows of the dataset:")
print(close_data.head(10))

# Plot original data
plt.figure(figsize=(14, 7))
plt.plot(close_data['Close'], label='Original Amazon Close Price', color='blue', alpha=0.7)
plt.title('Amazon Stock Closing Price Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Closing Price (USD)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Rolling Mean with window = 30 (monthly)
rolling_mean_30 = close_data['Close'].rolling(window=30).mean()
print("\nFirst 10 values of Rolling Mean (window=30):")
print(rolling_mean_30.head(10))

# Rolling Mean with window = 90 (quarterly)
rolling_mean_90 = close_data['Close'].rolling(window=90).mean()
print("\nFirst 20 values of Rolling Mean (window=90):")
print(rolling_mean_90.head(20))

# Plot Rolling Means
plt.figure(figsize=(14, 7))
plt.plot(close_data['Close'], label='Original', color='blue', alpha=0.5)
plt.plot(rolling_mean_30, label='Rolling Mean (30 Days)', color='orange', linewidth=2)
plt.plot(rolling_mean_90, label='Rolling Mean (90 Days)', color='green', linewidth=2)
plt.title('Moving Average (Rolling Mean) of Amazon Close Price', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Closing Price (USD)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Resample monthly and scale
close_monthly = close_data.resample('M').mean()
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(close_monthly.values.reshape(-1,1)).flatten(),
    index=close_monthly.index
)
scaled_data = scaled_data + 1

# Train-test split (80/20)
x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

# Exponential Smoothing Model
model = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
predictions = model.forecast(steps=len(test_data))

# Plot Train/Test/Predictions
ax = train_data.plot(figsize=(14,7), label="Train Data")
predictions.plot(ax=ax, label="Predictions")
test_data.plot(ax=ax, label="Test Data")
ax.set_title('Amazon Stock Price Forecast (Monthly)', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Scaled Closing Price', fontsize=12)
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# RMSE
rmse = np.sqrt(mean_squared_error(test_data, predictions))
print("\nRoot Mean Square Error (RMSE):", rmse)

# Variance and Mean
print("Variance:", np.sqrt(scaled_data.var()), "Mean:", scaled_data.mean())

# Future Forecast
model_full = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
future_predictions = model_full.forecast(steps=12)  # 12 months ahead

ax = scaled_data.plot(figsize=(14,7), label="Historical Monthly Close Price")
future_predictions.plot(ax=ax, label="Future Predictions")
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Scaled Closing Price', fontsize=12)
ax.set_title('Future Prediction of Amazon Monthly Closing Price', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()

```
## OUTPUT:
<img width="1390" height="690" alt="download" src="https://github.com/user-attachments/assets/461abb35-0637-4c8e-982d-0d1b9a04df34" />
<img width="1390" height="690" alt="download" src="https://github.com/user-attachments/assets/b69046e4-f6e3-4250-8ced-03325fcaf021" />
<img width="1389" height="690" alt="download" src="https://github.com/user-attachments/assets/f7351228-02ef-4d2b-9e4e-e4b6e8f198a0" />
<img width="528" height="52" alt="image" src="https://github.com/user-attachments/assets/9b8b88be-a240-4335-acb2-b71d5add0eb6" />


# Result:
The Exponential Smoothing model was successfully implemented for Amazon stock price forecasting, showing clear trend and seasonality patterns with reasonable predictive accuracy.
