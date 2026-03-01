# Databricks notebook source
# ==============================
# Final Wavelet-Based Forecasting Pipeline
# ==============================

# Install required libraries (uncomment if needed)
# !pip install pandas numpy matplotlib pywt scikit-learn

# Import libraries
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler














# COMMAND ----------

# ==============================
# Step 1: Load Data
# ==============================
df1 = pd.read_csv("NIFTY 50-01-04-2023-to-31-03-2024.csv")
df2 = pd.read_csv("NIFTY 50-01-04-2024-to-31-03-2025.csv")
df3 = pd.read_csv("NIFTY 50-01-04-2025-to-30-09-2025.csv")

train_df = pd.concat([df1, df2], ignore_index=True)
test_df = df3.copy()

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

train_df['Date'] = pd.to_datetime(train_df['Date'], dayfirst=True, errors='coerce')
test_df['Date'] = pd.to_datetime(test_df['Date'], dayfirst=True, errors='coerce')

train_df.sort_values('Date', inplace=True)
test_df.sort_values('Date', inplace=True)

# COMMAND ----------

# ==============================
# Step 2: Wavelet Denoising
# ==============================
def denoise_signal(signal, wavelet='db4', level=1):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    reconstructed = pywt.waverec(coeffs, wavelet)
    return reconstructed[:len(signal)]

train_df['Denoised_Close'] = denoise_signal(train_df['Close'].values)
test_df['Denoised_Close'] = denoise_signal(test_df['Close'].values)

# Plot Original vs Denoised
plt.figure(figsize=(12, 5))
plt.plot(train_df['Close'], label='Original Close')
plt.plot(train_df['Denoised_Close'], label='Denoised Close', linestyle='--')
plt.title('Original vs Denoised Close Price')
plt.legend()
plt.grid(True)
plt.show()


# COMMAND ----------


# ==============================
# Step 3: Add Advanced Features
# ==============================
def add_features(df):
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    return df




train_df = add_features(train_df)
test_df = add_features(test_df)

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)


# COMMAND ----------

# ==============================
# Step 4: Prepare Features
# ==============================
features = ['Open', 'High', 'Low', 'Close', 'Lag_1', 'Lag_2', 'MA_5', 'MA_10', 'EMA_5', 'EMA_10']
X_train = train_df[features].values
y_train = train_df['Denoised_Close'].values
X_test = test_df[features].values
y_test = test_df['Denoised_Close'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# COMMAND ----------

# ==============================
# Step 5: Train Models and Predict
# ==============================
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

results = {}
predictions = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    predictions[name] = preds
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# COMMAND ----------

# ==============================
# Step 6: Save Predictions to CSV
# ==============================
results_df = pd.DataFrame({
    'Date': test_df['Date'].values,
    'Actual_Close': y_test,
    'Predicted_LinearRegression': predictions['LinearRegression'],
    'Predicted_RandomForest': predictions['RandomForest'],
    'Predicted_GradientBoosting': predictions['GradientBoosting']
})
results_df.to_csv("nifty_predictions_final.csv", index=False)
print("✅ Predictions saved to nifty_predictions_final.csv")

# COMMAND ----------

# ==============================
# Step 7: Visualization
# ==============================
# Plot Prediction vs Actual for all models
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual', color='black')
plt.plot(predictions['LinearRegression'], label='Linear Regression', linestyle='--')
plt.plot(predictions['RandomForest'], label='Random Forest', linestyle='--')
plt.plot(predictions['GradientBoosting'], label='Gradient Boosting', linestyle='--')
plt.title('Prediction vs Actual Close Price')
plt.legend()
plt.grid(True)
plt.show()

# Plot Residuals for Best Model
best_model_name = min(results, key=lambda x: results[x]['RMSE'])
residuals = y_test - predictions[best_model_name]
plt.figure(figsize=(12, 5))
plt.plot(residuals, label=f'Residuals ({best_model_name})')
plt.title('Residuals of Best Model')
plt.grid(True)
plt.legend()
plt.show()

# Bar chart comparing model metrics
metrics_df = pd.DataFrame(results).T
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison (MAE, RMSE, R²)')
plt.grid(True)
plt.show()

# Print metrics
print("\nModel Performance Summary:")
print(metrics_df)