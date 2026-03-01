# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# --- Load and basic cleaning ---
files = [
    "NIFTY 50-01-04-2023-to-31-03-2024.csv",
    "NIFTY 50-01-04-2024-to-31-03-2025.csv",
    "NIFTY 50-01-04-2025-to-30-09-2025.csv"
]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').dropna(subset=['Date','Close']).reset_index(drop=True)

# --- Feature engineering ---
df['MA10'] = df['Close'].rolling(10).mean()
df['MA30'] = df['Close'].rolling(30).mean()
df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['EMA30'] = df['Close'].ewm(span=30, adjust=False).mean()

df['Daily_Change'] = df['Close'].diff()
df['Returns'] = df['Close'].pct_change()
df['Volatility10'] = df['Returns'].rolling(10).std()
df['Volatility20'] = df['Returns'].rolling(20).std()
df['High_Low_Range'] = df['High'] - df['Low']


df['Turnover_Norm'] = df['Turnover (₹ Cr)']  
df['Volume_Norm'] = df['Shares Traded']     # placeholder

df['Close_Lag1'] = df['Close'].shift(1)
df['Close_Lag2'] = df['Close'].shift(2)
df['Returns_Lag1'] = df['Returns'].shift(1)
df['Returns_Lag2'] = df['Returns'].shift(2)

# RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 1e-12)
df['RSI'] = 100 - (100 / (1 + rs))

# MACD
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Bollinger Bands Width
ma20 = df['Close'].rolling(20).mean()
std20 = df['Close'].rolling(20).std()
df['BB_Upper'] = ma20 + 2*std20
df['BB_Lower'] = ma20 - 2*std20
df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

# --- Parameters ---
SEQ_WINDOW = 30   
FFT_TOPK = 5

def compute_fft_features_window(arr, top_k=5):

    x = arr - np.mean(arr)
    fft_vals = np.fft.fft(x)
    fft_mag = np.abs(fft_vals)[:len(fft_vals)//2] 
    if fft_mag.sum() == 0:
        p = np.ones_like(fft_mag) / len(fft_mag)
    else:
        p = fft_mag / (fft_mag.sum() + 1e-12)
    dom_idx = np.argmax(fft_mag[1:]) + 1 if len(fft_mag) > 1 else 0
    dom_freq_norm = dom_idx / len(fft_mag)  
    spectral_energy = np.sum(fft_mag**2)
    spectral_entropy = -np.sum(p * np.log(p + 1e-12))
    topk_avg = np.mean(np.sort(fft_mag)[-top_k:]) if len(fft_mag) >= top_k else np.mean(fft_mag)
    return dom_freq_norm, spectral_energy, spectral_entropy, topk_avg

# initialize columns
df['FFT_DominantFreq'] = np.nan
df['FFT_Energy'] = np.nan
df['FFT_Entropy'] = np.nan
df['FFT_TopKAvg'] = np.nan

close_vals = df['Close'].values
for i in range(SEQ_WINDOW - 1, len(df)):
    window_vals = close_vals[i - SEQ_WINDOW + 1: i + 1]
    dom, energy, entropy, topk = compute_fft_features_window(window_vals, top_k=FFT_TOPK)
    df.at[i, 'FFT_DominantFreq'] = dom
    df.at[i, 'FFT_Energy'] = energy
    df.at[i, 'FFT_Entropy'] = entropy
    df.at[i, 'FFT_TopKAvg'] = topk


df = df.dropna().reset_index(drop=True)


train_df = df[(df['Date'] >= '2023-04-01') & (df['Date'] <= '2025-03-31')].copy().reset_index(drop=True)
test_df = df[(df['Date'] >= '2025-04-01') & (df['Date'] <= '2025-09-30')].copy().reset_index(drop=True)


turnover_max = train_df['Turnover (₹ Cr)'].max()
volume_max = train_df['Shares Traded'].max()
train_df['Turnover_Norm'] = train_df['Turnover (₹ Cr)'] / (turnover_max + 1e-12)
train_df['Volume_Norm'] = train_df['Shares Traded'] / (volume_max + 1e-12)
test_df['Turnover_Norm'] = test_df['Turnover (₹ Cr)'] / (turnover_max + 1e-12)
test_df['Volume_Norm'] = test_df['Shares Traded'] / (volume_max + 1e-12)

# --- Feature lists ---
features_with_fft = ['Close','FFT_DominantFreq','FFT_Energy','FFT_Entropy','FFT_TopKAvg',
                     'MA10','MA30','EMA10','EMA30','Daily_Change','Returns',
                     'Volatility10','Volatility20','High_Low_Range','Turnover_Norm','Volume_Norm',
                     'Close_Lag1','Close_Lag2','Returns_Lag1','Returns_Lag2',
                     'RSI','MACD','MACD_Signal','BB_Width']
features_without_fft = [f for f in features_with_fft if not f.startswith('FFT_')]


def create_sequences_from_array(arr, window=SEQ_WINDOW):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i-window:i])
        y.append(arr[i, 0])  # target is first column (Close)
    return np.array(X), np.array(y)


def build_lstm(input_shape, bidirectional=False):
    model = Sequential()
    if bidirectional:
        model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape))
    else:
        model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_and_evaluate(train_df, test_df, features, label, bidirectional=False, epochs=50):
   
    scaler = MinMaxScaler()
    train_feats = train_df[features].values
    test_feats = test_df[features].values
    scaler.fit(train_feats)
    train_scaled = scaler.transform(train_feats)
    test_scaled = scaler.transform(test_feats)

    
    X_train, y_train = create_sequences_from_array(train_scaled, window=SEQ_WINDOW)
    X_test, y_test = create_sequences_from_array(test_scaled, window=SEQ_WINDOW)

    
    model = build_lstm((X_train.shape[1], X_train.shape[2]), bidirectional=bidirectional)
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1,
              validation_split=0.1, callbacks=[es, lr])

    
    pred_test_scaled = model.predict(X_test).reshape(-1, 1)

    
    pad_pred = np.hstack([pred_test_scaled, np.zeros((pred_test_scaled.shape[0], len(features)-1))])
    pred_test_rescaled = scaler.inverse_transform(pad_pred)[:, 0]

    pad_actual = np.hstack([y_test.reshape(-1,1), np.zeros((y_test.shape[0], len(features)-1))])
    actual_test_rescaled = scaler.inverse_transform(pad_actual)[:, 0]

    # Metrics
    mae = mean_absolute_error(actual_test_rescaled, pred_test_rescaled)
    rmse = np.sqrt(mean_squared_error(actual_test_rescaled, pred_test_rescaled))
    
    mape = np.mean(np.abs((actual_test_rescaled - pred_test_rescaled) / (actual_test_rescaled + 1e-12))) * 100
    directional_acc = np.mean(
        np.sign(np.diff(actual_test_rescaled)) == np.sign(np.diff(pred_test_rescaled))
    )

    
    dates_plot = test_df['Date'].values[SEQ_WINDOW:]
    plt.figure(figsize=(12,6))
    plt.plot(dates_plot, actual_test_rescaled, label='Actual Close Price')
    plt.plot(dates_plot, pred_test_rescaled, label=f'Forecasted ({label})')
    plt.title(f'NIFTY 50 Forecasting ({label})')
    plt.xlabel('Date'); plt.ylabel('Close Price'); plt.legend(); plt.show()

    print(f"{label} - Test MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, Directional Accuracy: {directional_acc:.2f}")
    return mae, rmse, mape, directional_acc


mae_fft, rmse_fft, mape_fft, diracc_fft = train_and_evaluate(train_df, test_df, features_with_fft, "With FFT", bidirectional=True, epochs=50)
mae_nofft, rmse_nofft, mape_nofft, diracc_nofft = train_and_evaluate(train_df, test_df, features_without_fft, "Without FFT", bidirectional=True, epochs=50)

print("Comparison of LSTM:")
print(f"With FFT -> MAE: {mae_fft:.2f}, RMSE: {rmse_fft:.2f}, MAPE: {mape_fft:.2f}%, DirAcc: {diracc_fft:.2f}")
print(f"Without FFT -> MAE: {mae_nofft:.2f}, RMSE: {rmse_nofft:.2f}, MAPE: {mape_nofft:.2f}%, DirAcc: {diracc_nofft:.2f}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import xgboost as xgb
import joblib
import random
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def create_flat_sequences_from_array(arr, window=SEQ_WINDOW):

    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i-window:i].reshape(-1))  
        y.append(arr[i, 0])                     
    return np.array(X), np.array(y)

def train_and_eval_ml(train_df, test_df, features, label, models_dict, tune=False, n_jobs=-1):
   
    scaler = MinMaxScaler()
    train_feats = train_df[features].values
    test_feats = test_df[features].values
    scaler.fit(train_feats)                     
    train_scaled = scaler.transform(train_feats)
    test_scaled = scaler.transform(test_feats)

    
    X_train, y_train = create_flat_sequences_from_array(train_scaled, window=SEQ_WINDOW)
    X_test, y_test = create_flat_sequences_from_array(test_scaled, window=SEQ_WINDOW)

    results = {}
    for name, model in models_dict.items():
        print(f"\nTraining {name} ...")
        clf = model

        if tune and name in ['RandomForest','XGBoost']:
            
            tscv = TimeSeriesSplit(n_splits=3)
            if name == 'RandomForest':
                param_dist = {
                    'n_estimators': [100, 200, 400],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2,5,10]
                }
            else:  # XGBoost
                param_dist = {
                    'n_estimators': [100,200,400],
                    'max_depth': [3,5,8],
                    'learning_rate': [0.01,0.05,0.1],
                    'subsample': [0.6,0.8,1.0]
                }
            rs = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10,
                                    cv=tscv, scoring='neg_mean_squared_error',
                                    random_state=SEED, n_jobs=n_jobs, verbose=0)
            rs.fit(X_train, y_train)
            clf = rs.best_estimator_
            print(f" Best params: {rs.best_params_}")

        else:
            clf.fit(X_train, y_train)

       
        pred_test_scaled = clf.predict(X_test).reshape(-1, 1)

        
        pad_pred = np.hstack([pred_test_scaled, np.zeros((pred_test_scaled.shape[0], len(features)-1))])
        pred_test_rescaled = scaler.inverse_transform(pad_pred)[:, 0]

        pad_actual = np.hstack([y_test.reshape(-1,1), np.zeros((y_test.shape[0], len(features)-1))])
        actual_test_rescaled = scaler.inverse_transform(pad_actual)[:, 0]

        # Metrics
        mae = mean_absolute_error(actual_test_rescaled, pred_test_rescaled)
        rmse = np.sqrt(mean_squared_error(actual_test_rescaled, pred_test_rescaled))
        mape = np.mean(np.abs((actual_test_rescaled - pred_test_rescaled) / (actual_test_rescaled + 1e-12))) * 100
        directional_acc = np.mean(
            np.sign(np.diff(actual_test_rescaled)) == np.sign(np.diff(pred_test_rescaled))
        )

        # Plot
        dates_plot = test_df['Date'].values[SEQ_WINDOW:]
        plt.figure(figsize=(12,5))
        plt.plot(dates_plot, actual_test_rescaled, label='Actual Close Price')
        plt.plot(dates_plot, pred_test_rescaled, label=f'Forecasted ({label} - {name})')
        plt.title(f'NIFTY 50 Forecasting ({label} - {name})')
        plt.xlabel('Date'); plt.ylabel('Close Price'); plt.legend(); plt.show()

        print(f"{label} | {name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, DirAcc: {directional_acc:.2f}")

        results[name] = {
            'model': clf,
            'mae': mae, 'rmse': rmse, 'mape': mape, 'diracc': directional_acc,
            'pred': pred_test_rescaled, 'actual': actual_test_rescaled
        }

    return results


models = {
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, random_state=SEED, n_jobs=-1, objective='reg:squarederror'),
    'Lasso': Lasso(alpha=0.001, random_state=SEED, max_iter=5000),
    'Ridge': Ridge(alpha=1.0, random_state=SEED)
}


res_with_fft = train_and_eval_ml(train_df, test_df, features_with_fft, "With FFT", models, tune=False)
res_no_fft = train_and_eval_ml(train_df, test_df, features_without_fft, "Without FFT", models, tune=False)


print("\nSummary (With FFT):")
for name, r in res_with_fft.items():
    print(f"{name}: MAE={r['mae']:.2f}, RMSE={r['rmse']:.2f}, MAPE={r['mape']:.2f}%, DirAcc={r['diracc']:.2f}")

print("\nSummary (Without FFT):")
for name, r in res_no_fft.items():
    print(f"{name}: MAE={r['mae']:.2f}, RMSE={r['rmse']:.2f}, MAPE={r['mape']:.2f}%, DirAcc={r['diracc']:.2f}")


# COMMAND ----------

