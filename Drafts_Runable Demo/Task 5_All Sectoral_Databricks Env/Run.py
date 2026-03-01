# Databricks notebook source
# %pip install xgboost scikit-learn tensorflow pyarrow scipy pandas numpy matplotlib seaborn -q
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from scipy.fft import fft
print("Step 1: Libraries imported successfully")
csv_folder_path = "/Workspace/Users/akshatp@ida.tcsapps.com/Drafts/csv"
SECTORS = ['NIFTY 50', 'NIFTY BANK', 'NIFTY IT', 'NIFTY AUTO', 
           'NIFTY ENERGY', 'NIFTY METAL', 'NIFTY MEDIA']
print(f"Step 2: Processing {len(SECTORS)} sectors from {csv_folder_path}")
def clean_column_names(df):
    if df is None or df.empty:
        return df
    column_mapping = {}
    for col in df.columns:
        clean_col = col.strip()
        clean_col = clean_col.replace('₹', 'Rs').replace('(', '').replace(')', '').replace(' ', '_')
        clean_col = ''.join(c for c in clean_col if c.isalnum() or c == '_')
        column_mapping[col] = clean_col
    return df.rename(columns=column_mapping)
def load_sector_data(sector_name):
    try:
        all_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
        sector_files = [f for f in all_files if sector_name.upper() in f.upper()]
        
        if not sector_files:
            return None, None
        
        train_dfs = []
        forecast_dfs = []
        
        for file in sorted(sector_files):
            file_path = os.path.join(csv_folder_path, file)
            
            try:
                df = pd.read_csv(file_path)
                df = clean_column_names(df)
                
                required = ['Date', 'Open', 'High', 'Low', 'Close']
                if not all(col in df.columns for col in required):
                    continue
                
                df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
                df = df.sort_values('Date').reset_index(drop=True)
                
                if '2023' in file or '2024' in file or ('2025' in file and 'to-31-03-2025' in file.lower()):
                    train_dfs.append(df)
                else:
                    forecast_dfs.append(df)
                    
            except:
                continue
        
        train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else None
        forecast_df = pd.concat(forecast_dfs, ignore_index=True) if forecast_dfs else None
        
        if train_df is not None:
            train_df = train_df.drop_duplicates(subset=['Date']).sort_values('Date')
            train_df = train_df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if forecast_df is not None and not forecast_df.empty:
            forecast_df = forecast_df.drop_duplicates(subset=['Date']).sort_values('Date')
            forecast_df = forecast_df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        return train_df, forecast_df
    
    except:
        return None, None
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
def calculate_macd(prices, slow=26, fast=12, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal
def create_features(df, use_fft=False):
    if df is None or len(df) < 30:
        return None
    
    df = df.copy()
    
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['High_Low_Spread'] = df['High'] - df['Low']
    df['Close_Open_Spread'] = df['Close'] - df['Open']
    
    for window in [5, 10, 20]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    df['Volatility'] = df['Returns'].rolling(window=10).std()
    
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    
    df['Close_To_High'] = df['Close'] / df['High']
    df['Close_To_Low'] = df['Close'] / df['Low']
    
    df['Volatility_Clustering'] = df['Returns'].rolling(window=5).std() / df['Returns'].rolling(window=20).std()
    
    if 'Date' in df.columns:
        df['Day_of_Week'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Week_of_Year'] = df['Date'].dt.isocalendar().week
    
    for lag in [1, 2, 3, 5]:
        df[f'Return_Lag_{lag}'] = df['Returns'].shift(lag)
    
    for window in [5, 10, 20]:
        df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Rolling_Std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'Rolling_Min_{window}'] = df['Close'].rolling(window=window).min()
        df[f'Rolling_Max_{window}'] = df['Close'].rolling(window=window).max()
    
    df['Price_Change_Pct'] = df['Close'].pct_change() * 100
    
    df['TR1'] = abs(df['High'] - df['Low'])
    df['TR2'] = abs(df['High'] - df['Close'].shift())
    df['TR3'] = abs(df['Low'] - df['Close'].shift())
    df['True_Range'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
    df = df.drop(['TR1', 'TR2', 'TR3'], axis=1)
    
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()
    
    if 'Shares_Traded' in df.columns:
        df['Volume_MA_5'] = df['Shares_Traded'].rolling(window=5).mean()
        df['Volume_Ratio'] = df['Shares_Traded'] / df['Shares_Traded'].rolling(window=20).mean()
        for lag in [1, 2, 3]:
            df[f'Volume_Lag_{lag}'] = df['Shares_Traded'].shift(lag)
    
    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    
    if use_fft:
        try:
            close_values = df['Close'].dropna().values
            if len(close_values) > 10:
                close_fft = fft(close_values)
                magnitude = np.abs(close_fft)
                if len(magnitude) > 5:
                    for i in range(1, 6):
                        df[f'FFT_Magnitude_{i}'] = magnitude[i]
        except:
            for i in range(1, 6):
                df[f'FFT_Magnitude_{i}'] = 0
    
    df['Target'] = df['Close'].shift(-1)
    
    return df.dropna()
def calculate_metrics(y_true, y_pred):
    metrics = {}
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    mask = y_true != 0
    if mask.any():
        metrics['MAPE'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    if len(y_true) > 1 and len(y_pred) > 1:
        true_dir = np.diff(y_true) > 0
        pred_dir = np.diff(y_pred) > 0
        if len(true_dir) > 0:
            metrics['Directional_Accuracy'] = np.mean(true_dir == pred_dir) * 100
    
    metrics['R2'] = r2_score(y_true, y_pred)
    
    return metrics
def prepare_lstm_data(X, y, time_steps=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)
def create_lstm_model(input_shape, with_fft=False):
    model = Sequential()
    
    if with_fft:
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape, 
                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(32, return_sequences=False, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
    else:
        model.add(LSTM(100, return_sequences=True, input_shape=input_shape, 
                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(50, return_sequences=False, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
    
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
    
    return model
def debug_lstm_predictions(y_true, y_pred, model_name):
    print(f"\n🔍 Debugging {model_name}:")
    print(f"  True values range: {y_true.min():.2f} to {y_true.max():.2f}")
    print(f"  Pred values range: {y_pred.min():.2f} to {y_pred.max():.2f}")
    print(f"  Mean true: {y_true.mean():.2f}, Mean pred: {y_pred.mean():.2f}")
    print(f"  Std true: {y_true.std():.2f}, Std pred: {y_pred.std():.2f}")
    
    threshold = y_true.std() * 3
    outliers = np.abs(y_pred - y_true) > threshold
    print(f"  Outliers: {outliers.sum()} ({outliers.sum()/len(y_true)*100:.1f}%)")
class ChronosModel:
    def __init__(self):
        self.models = []
        self.weights = []
        
    def add_model(self, model, weight):
        self.models.append(model)
        self.weights.append(weight)
        
    def fit(self, X, y):
        for model in self.models:
            if hasattr(model, 'fit'):
                try:
                    model.fit(X, y)
                except:
                    continue
                
    def predict(self, X):
        if not self.models:
            return np.zeros(X.shape[0])
        
        predictions = []
        valid_weights = []
        
        for model, weight in zip(self.models, self.weights):
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    # Check if predictions are reasonable
                    if not np.any(np.isnan(pred)) and not np.any(np.isinf(pred)):
                        predictions.append(pred)
                        valid_weights.append(weight)
            except:
                continue
        
        if not predictions:
            return np.zeros(X.shape[0])
        
        predictions = np.array(predictions)
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()
        
        weighted_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_pred += pred * valid_weights[i]
            
        return weighted_pred
    
    def optimize_weights(self, X_val, y_val, base_models):
        predictions = []
        model_list = []
        
        for model in base_models:
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X_val)
                    # Check prediction quality
                    if len(pred) == len(y_val) and not np.any(np.isnan(pred)):
                        predictions.append(pred)
                        model_list.append(model)
            except:
                continue
        
        if len(predictions) < 2:
            # If less than 2 valid models, use equal weights
            self.models = base_models
            self.weights = [1.0/len(base_models)] * len(base_models)
            return
        
        weights = []
        for pred in predictions:
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            # Use inverse RMSE (lower RMSE = higher weight)
            weight = 1.0 / (rmse + 1e-10)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        self.models = model_list
        self.weights = list(weights)
def train_sector_models(sector_name):
    print(f"\n{'='*80}")
    print(f"Processing: {sector_name}")
    print(f"{'='*80}")
    
    train_df, forecast_df = load_sector_data(sector_name)
    if train_df is None or len(train_df) < 100:
        return None, None, None
    
    print(f"Training data: {len(train_df)} rows")
    if forecast_df is not None:
        print(f"Forecast data: {len(forecast_df)} rows")
    
    df_no_fft = create_features(train_df, use_fft=False)
    df_with_fft = create_features(train_df, use_fft=True)
    
    if df_no_fft is None:
        return None, None, None
    
    exclude_cols = ['Date', 'Target', 'Close', 'Returns', 'Log_Returns']
    base_features = [col for col in df_no_fft.columns if col not in exclude_cols and not col.startswith('FFT')]
    
    fft_features = base_features.copy()
    fft_cols = [col for col in df_with_fft.columns if 'FFT' in col]
    fft_features.extend(fft_cols)
    
    X = df_no_fft[base_features].values
    X_fft = df_with_fft[fft_features].values
    y = df_no_fft['Target'].values
    
    tscv = TimeSeriesSplit(n_splits=5)
    split_idx = list(tscv.split(X))[-1][1][0]
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    X_train_fft, X_test_fft = X_fft[:split_idx], X_fft[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_fft = StandardScaler()
    X_train_fft_scaled = scaler_fft.fit_transform(X_train_fft)
    X_test_fft_scaled = scaler_fft.transform(X_test_fft)
    
    results = {}
    
    print(f"\nTraining RandomForest...")
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, 
                                     random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    metrics_rf = calculate_metrics(y_test, y_pred_rf)
    results['RandomForest'] = {'model': rf_model, 'metrics': metrics_rf, 
                              'scaler': scaler, 'features': base_features}
    print(f"RandomForest - R²: {metrics_rf['R2']:.3f}, RMSE: {metrics_rf['RMSE']:.2f}")
    
    print(f"Training XGBoost...")
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, 
                            max_depth=6, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    metrics_xgb = calculate_metrics(y_test, y_pred_xgb)
    results['XGBoost'] = {'model': xgb_model, 'metrics': metrics_xgb, 
                         'scaler': scaler, 'features': base_features}
    print(f"XGBoost - R²: {metrics_xgb['R2']:.3f}, RMSE: {metrics_xgb['RMSE']:.2f}")
    
    print(f"Training Lasso...")
    lasso_model = Lasso(alpha=0.001, random_state=42, max_iter=10000)
    lasso_model.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso_model.predict(X_test_scaled)
    metrics_lasso = calculate_metrics(y_test, y_pred_lasso)
    results['Lasso'] = {'model': lasso_model, 'metrics': metrics_lasso, 
                       'scaler': scaler, 'features': base_features}
    print(f"Lasso - R²: {metrics_lasso['R2']:.3f}, RMSE: {metrics_lasso['RMSE']:.2f}")
    
    print(f"Training Ridge...")
    ridge_model = Ridge(alpha=0.1, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    metrics_ridge = calculate_metrics(y_test, y_pred_ridge)
    results['Ridge'] = {'model': ridge_model, 'metrics': metrics_ridge, 
                       'scaler': scaler, 'features': base_features}
    print(f"Ridge - R²: {metrics_ridge['R2']:.3f}, RMSE: {metrics_ridge['RMSE']:.2f}")
    print(f"Training LSTM...")
    time_steps = 10
    lstm_scaler = StandardScaler()
    X_train_lstm_full = np.column_stack([X_train_scaled, y_train.reshape(-1, 1)])
    lstm_scaler.fit(X_train_lstm_full)
    X_train_lstm_scaled = lstm_scaler.transform(np.column_stack([X_train_scaled, y_train.reshape(-1, 1)]))
    X_test_lstm_scaled = lstm_scaler.transform(np.column_stack([X_test_scaled, y_test.reshape(-1, 1)]))
    X_train_lstm_features = X_train_lstm_scaled[:, :-1]
    y_train_lstm_target = X_train_lstm_scaled[:, -1]
    X_test_lstm_features = X_test_lstm_scaled[:, :-1]
    y_test_lstm_target = X_test_lstm_scaled[:, -1]
    def prepare_lstm_sequences_correctly(features, target, time_steps):
        X_seq, y_seq = [], []
        for i in range(time_steps, len(features)):
            X_seq.append(features[i-time_steps:i])
            y_seq.append(target[i])  # Predict current value
        return np.array(X_seq), np.array(y_seq)
    X_train_seq, y_train_seq = prepare_lstm_sequences_correctly(
        X_train_lstm_features, y_train_lstm_target, time_steps
    )
    X_test_seq, y_test_seq = prepare_lstm_sequences_correctly(
        X_test_lstm_features, y_test_lstm_target, time_steps
    )
    y_test_actual = y_test[time_steps:]
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_steps, X_train_seq.shape[2]), 
            recurrent_dropout=0.1),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32, return_sequences=False, recurrent_dropout=0.1),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.0005, clipvalue=1.0)
    lstm_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True, 
        min_delta=0.0001
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=10, 
        min_lr=1e-7
    )
    history = lstm_model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    y_pred_seq = lstm_model.predict(X_test_seq, verbose=0).flatten()
    y_pred_original = []
    for i, pred in enumerate(y_pred_seq):
        # Create a dummy row with zeros for features and prediction for target
        dummy_row = np.zeros(X_train_lstm_scaled.shape[1])
        dummy_row[-1] = pred
        # Inverse transform
        inverse_row = lstm_scaler.inverse_transform(dummy_row.reshape(1, -1))
        y_pred_original.append(inverse_row[0, -1])
    y_pred_lstm = np.array(y_pred_original)
    min_len = min(len(y_pred_lstm), len(y_test_actual))
    if min_len > 0:
        y_pred_lstm = y_pred_lstm[:min_len]
        y_test_actual = y_test_actual[:min_len]
        
        metrics_lstm = calculate_metrics(y_test_actual, y_pred_lstm)
        results['LSTM'] = {
            'model': lstm_model,
            'metrics': metrics_lstm,
            'scaler': lstm_scaler,
            'features': base_features,
            'time_steps': time_steps
        }
        print(f"LSTM - R²: {metrics_lstm['R2']:.3f}, RMSE: {metrics_lstm['RMSE']:.2f}")
    else:
        print(f"LSTM - Not enough data for evaluation")
    print(f"Training LSTM with FFT...")
    lstm_fft_scaler = StandardScaler()
    X_train_fft_full = np.column_stack([X_train_fft_scaled, y_train.reshape(-1, 1)])
    lstm_fft_scaler.fit(X_train_fft_full)
    X_train_fft_lstm_scaled = lstm_fft_scaler.transform(
        np.column_stack([X_train_fft_scaled, y_train.reshape(-1, 1)])
    )
    X_test_fft_lstm_scaled = lstm_fft_scaler.transform(
        np.column_stack([X_test_fft_scaled, y_test.reshape(-1, 1)])
    )
    X_train_fft_features = X_train_fft_lstm_scaled[:, :-1]
    y_train_fft_target = X_train_fft_lstm_scaled[:, -1]
    X_test_fft_features = X_test_fft_lstm_scaled[:, :-1]
    y_test_fft_target = X_test_fft_lstm_scaled[:, -1]
    X_train_fft_seq, y_train_fft_seq = prepare_lstm_sequences_correctly(
        X_train_fft_features, y_train_fft_target, time_steps
    )
    X_test_fft_seq, y_test_fft_seq = prepare_lstm_sequences_correctly(
        X_test_fft_features, y_test_fft_target, time_steps
    )
    y_test_fft_actual = y_test[time_steps:]
    lstm_fft_model = Sequential([
        LSTM(80, return_sequences=True, input_shape=(time_steps, X_train_fft_seq.shape[2])),
        Dropout(0.2),
        LSTM(40, return_sequences=False),
        Dropout(0.2),
        Dense(20, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)
    ])
    lstm_fft_model.compile(optimizer=Adam(learning_rate=0.0003), loss='mse', metrics=['mae'])
    history_fft = lstm_fft_model.fit(
        X_train_fft_seq, y_train_fft_seq,
        epochs=80,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    y_pred_fft_seq = lstm_fft_model.predict(X_test_fft_seq, verbose=0).flatten()
    y_pred_fft_original = []
    for i, pred in enumerate(y_pred_fft_seq):
        dummy_row = np.zeros(X_train_fft_lstm_scaled.shape[1])
        dummy_row[-1] = pred
        inverse_row = lstm_fft_scaler.inverse_transform(dummy_row.reshape(1, -1))
        y_pred_fft_original.append(inverse_row[0, -1])
    y_pred_lstm_fft = np.array(y_pred_fft_original)
    min_len = min(len(y_pred_lstm_fft), len(y_test_fft_actual))
    if min_len > 0:
        y_pred_lstm_fft = y_pred_lstm_fft[:min_len]
        y_test_fft_actual = y_test_fft_actual[:min_len]
        
        debug_lstm_predictions(y_test_actual, y_pred_lstm, "LSTM")
        metrics_lstm_fft = calculate_metrics(y_test_fft_actual, y_pred_lstm_fft)
        results['LSTM_FFT'] = {
            'model': lstm_fft_model,
            'metrics': metrics_lstm_fft,
            'scaler': lstm_fft_scaler,
            'features': fft_features,
            'time_steps': time_steps
        }
        print(f"LSTM+FFT - R²: {metrics_lstm_fft['R2']:.3f}, RMSE: {metrics_lstm_fft['RMSE']:.2f}")
    else:
        print(f"LSTM+FFT - Not enough data for evaluation")
    
    print(f"Training Chronos ensemble...")
    chronos = ChronosModel()
    base_models = [rf_model, xgb_model, lasso_model, ridge_model]
    for model in base_models:
        chronos.add_model(model, 1.0)
    
    chronos.optimize_weights(X_test_scaled, y_test, base_models)
    y_pred_chronos = chronos.predict(X_test_scaled)
    metrics_chronos = calculate_metrics(y_test, y_pred_chronos)
    results['Chronos'] = {'model': chronos, 'metrics': metrics_chronos}
    print(f"Chronos - R²: {metrics_chronos['R2']:.3f}, RMSE: {metrics_chronos['RMSE']:.2f}")
    
    print(f"\nModel Performance for {sector_name}:")
    print("-" * 100)
    print(f"{'Model':<15} {'R²':<10} {'RMSE':<15} {'MAE':<15} {'MAPE':<15} {'Dir Acc %':<12}")
    print("-" * 100)
    
    for model_name in ['LSTM_FFT', 'LSTM', 'XGBoost', 'RandomForest', 'Ridge', 'Lasso', 'Chronos']:
        if model_name in results:
            metrics = results[model_name]['metrics']
            mapes = metrics.get('MAPE', 'N/A')
            dir_acc = metrics.get('Directional_Accuracy', 'N/A')
            print(f"{model_name:<15} {metrics['R2']:<10.3f} ₹{metrics['RMSE']:<13.2f} "
                  f"₹{metrics['MAE']:<13.2f} {mapes if isinstance(mapes, str) else f'{mapes:.2f}%':<14} "
                  f"{dir_acc if isinstance(dir_acc, str) else f'{dir_acc:.1f}%':<11}")
    
    print("-" * 100)
    
    best_model_name = None
    best_r2 = -float('inf')
    for model_name, result in results.items():
        if result['metrics']['R2'] > best_r2:
            best_r2 = result['metrics']['R2']
            best_model_name = model_name
    
    print(f"\n🏆 Best model for {sector_name}: {best_model_name} (R²: {best_r2:.3f})")
    
    plot_model_comparison(sector_name, results)
    
    return results, best_model_name, forecast_df
def plot_model_comparison(sector_name, results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{sector_name} - Model Performance Comparison', fontsize=16, fontweight='bold')
    
    model_names = []
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    dir_acc_scores = []
    
    for model_name in ['LSTM_FFT', 'LSTM', 'XGBoost', 'RandomForest', 'Ridge', 'Lasso', 'Chronos']:
        if model_name in results:
            metrics = results[model_name]['metrics']
            model_names.append(model_name)
            r2_scores.append(metrics['R2'])
            rmse_scores.append(metrics['RMSE'])
            mae_scores.append(metrics['MAE'])
            mape = metrics.get('MAPE', 0)
            mape_scores.append(mape if not isinstance(mape, str) else 0)
            dir_acc = metrics.get('Directional_Accuracy', 0)
            dir_acc_scores.append(dir_acc if not isinstance(dir_acc, str) else 0)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    axes[0, 0].bar(model_names, r2_scores, color=colors)
    axes[0, 0].set_title('R² Score Comparison')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].bar(model_names, rmse_scores, color=colors)
    axes[0, 1].set_title('RMSE Comparison')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    axes[0, 2].bar(model_names, mae_scores, color=colors)
    axes[0, 2].set_title('MAE Comparison')
    axes[0, 2].set_ylabel('MAE')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    axes[1, 0].bar(model_names, mape_scores, color=colors)
    axes[1, 0].set_title('MAPE Comparison')
    axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 1].bar(model_names, dir_acc_scores, color=colors)
    axes[1, 1].set_title('Directional Accuracy Comparison')
    axes[1, 1].set_ylabel('Directional Accuracy (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    x_pos = np.arange(len(model_names))
    axes[1, 2].plot(x_pos, r2_scores, 'o-', label='R²', linewidth=2, markersize=8)
    axes[1, 2].plot(x_pos, np.array(dir_acc_scores)/100, 's-', label='Dir Acc (scaled)', linewidth=2, markersize=8)
    axes[1, 2].set_title('R² vs Directional Accuracy')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(model_names, rotation=45)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"/Workspace/Users/akshatp@ida.tcsapps.com/Drafts/plots_{sector_name.replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Plot saved: {plot_path}")
def generate_forecasts(sector_name, train_results, forecast_df):
    if forecast_df is None or forecast_df.empty:
        print(f"No forecast data for {sector_name}")
        return None
    
    best_model_name = None
    best_result = None
    for model_name, result in train_results.items():
        if model_name != 'Chronos' and 'model' in result and result['model'] is not None:
            if best_model_name is None or result['metrics']['R2'] > best_result['metrics']['R2']:
                best_model_name = model_name
                best_result = result
    
    if not best_result:
        print(f"No suitable model found for forecasting {sector_name}")
        return None
    
    print(f"Generating forecasts for {sector_name} using {best_model_name}...")
    
    use_fft = 'FFT' in best_model_name
    forecast_features = create_features(forecast_df, use_fft=use_fft)
    if forecast_features is None:
        return None
    
    feature_cols = best_result.get('features', [])
    available_cols = [col for col in feature_cols if col in forecast_features.columns]
    
    if len(available_cols) == 0:
        return None
    
    X_forecast = forecast_features[available_cols].values
    X_forecast_scaled = best_result['scaler'].transform(X_forecast)
    
    try:
        if 'LSTM' in best_model_name:
            time_steps = best_result.get('time_steps', 10)
            
            # Use the specialized LSTM forecasting function
            predictions, start_idx = generate_lstm_forecast(
                best_result, X_forecast_scaled, forecast_features, time_steps
            )
            
            if predictions is None:
                print(f"Failed to generate LSTM forecasts for {sector_name}")
                return None
            
            # Align dates and actual values with predictions
            forecast_dates = forecast_features['Date'].iloc[start_idx:start_idx + len(predictions)]
            actual_values = forecast_features['Close'].iloc[start_idx:start_idx + len(predictions)].values
            
        else:
            # Non-LSTM models
            predictions = best_result['model'].predict(X_forecast_scaled)
            forecast_dates = forecast_features['Date']
            actual_values = forecast_features['Close'].values
        
        min_len = min(len(predictions), len(actual_values), len(forecast_dates))
        predictions = predictions[:min_len]
        actual_values = actual_values[:min_len]
        forecast_dates = forecast_dates.iloc[:min_len]
        
        forecast_results = pd.DataFrame({
            'Date': forecast_dates.reset_index(drop=True),
            'Actual_Close': actual_values,
            'Predicted_Close': predictions,
            'Model': best_model_name,
            'Sector': sector_name,
            'Error': actual_values - predictions,
            'Absolute_Error': np.abs(actual_values - predictions)
        })
        
        if (actual_values != 0).any():
            forecast_results['Percentage_Error'] = (forecast_results['Error'] / actual_values) * 100
        
        output_path = f"/Workspace/Users/akshatp@ida.tcsapps.com/Drafts/final_forecasts_{sector_name.replace(' ', '_')}.csv"
        forecast_results.to_csv(output_path, index=False)
        
        plot_forecasts(sector_name, forecast_results, best_model_name)
        
        forecast_rmse = np.sqrt(np.mean(forecast_results['Error']**2))
        forecast_mae = np.mean(forecast_results['Absolute_Error'])
        
        print(f"✅ Forecasts saved: {output_path}")
        print(f"   Period: {forecast_results['Date'].min().date()} to {forecast_results['Date'].max().date()}")
        print(f"   Rows: {len(forecast_results)}")
        print(f"   Avg Prediction: ₹{forecast_results['Predicted_Close'].mean():.2f}")
        print(f"   Forecast RMSE: ₹{forecast_rmse:.2f}")
        print(f"   Forecast MAE: ₹{forecast_mae:.2f}")
        
        return forecast_results
        
    except Exception as e:
        print(f"Forecasting error: {str(e)}")
        return None
    
def generate_lstm_forecast(model_result, X_forecast_scaled, forecast_features, time_steps=10):
    try:
        model = model_result['model']
        scaler = model_result['scaler']
        
        if len(X_forecast_scaled) < time_steps:
            print(f"Not enough data for LSTM forecasting. Need {time_steps} time steps, have {len(X_forecast_scaled)}")
            return None, None
        
        predictions = []
        
        for i in range(len(X_forecast_scaled) - time_steps + 1):
            try:
                sequence = X_forecast_scaled[i:i + time_steps]
                
                sequence_with_dummy = np.zeros((time_steps, sequence.shape[1] + 1))
                sequence_with_dummy[:, :-1] = sequence
                
                sequence_scaled = scaler.transform(sequence_with_dummy)
                sequence_features = sequence_scaled[:, :-1]
                
                sequence_reshaped = sequence_features.reshape(1, time_steps, -1)
                
                pred_scaled = model.predict(sequence_reshaped, verbose=0)[0][0]
                
                dummy_row = np.zeros(sequence_with_dummy.shape[1])
                dummy_row[-1] = pred_scaled
                inverse_row = scaler.inverse_transform(dummy_row.reshape(1, -1))
                predictions.append(inverse_row[0, -1])
                
            except Exception as e:
                print(f"Error in sequence {i}: {str(e)}")
                continue
        
        if not predictions:
            return None, None
        
        predictions = np.array(predictions)
        start_idx = time_steps - 1
        
        return predictions, start_idx
        
    except Exception as e:
        print(f"LSTM forecast error: {str(e)}")
        return None, None
def plot_forecasts(sector_name, forecast_results, model_name):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{sector_name} - {model_name} Forecasts', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(forecast_results['Date'], forecast_results['Actual_Close'], 
                   label='Actual', linewidth=2, color='blue')
    axes[0, 0].plot(forecast_results['Date'], forecast_results['Predicted_Close'], 
                   label='Predicted', linewidth=2, color='red', linestyle='--')
    axes[0, 0].set_title('Actual vs Predicted Prices')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Close Price')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].scatter(forecast_results['Actual_Close'], forecast_results['Predicted_Close'], 
                      alpha=0.5, s=30)
    min_val = forecast_results[['Actual_Close', 'Predicted_Close']].min().min()
    max_val = forecast_results[['Actual_Close', 'Predicted_Close']].max().max()
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0, 1].set_title('Actual vs Predicted Scatter')
    axes[0, 1].set_xlabel('Actual Close')
    axes[0, 1].set_ylabel('Predicted Close')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(forecast_results['Date'], forecast_results['Error'], 
                   linewidth=1.5, color='green')
    axes[1, 0].axhline(y=0, color='r', linestyle='-', linewidth=1)
    axes[1, 0].set_title('Forecast Errors Over Time')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Error (Actual - Predicted)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 1].hist(forecast_results['Error'], bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='-', linewidth=2)
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].set_xlabel('Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    forecast_plot_path = f"/Workspace/Users/akshatp@ida.tcsapps.com/Drafts/forecast_plots_{sector_name.replace(' ', '_')}.png"
    plt.savefig(forecast_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Forecast plot saved: {forecast_plot_path}")
print("\nStep 3: Starting ModelOps Pipeline Execution")
print("="*80)
all_results = {}
all_forecasts = {}
for sector in SECTORS:
    try:
        results, best_model, forecast_df = train_sector_models(sector)
        if results:
            all_results[sector] = results
            
            forecasts = generate_forecasts(sector, results, forecast_df)
            if forecasts is not None:
                all_forecasts[sector] = forecasts
    except Exception as e:
        print(f"Error processing {sector}: {str(e)}")
        continue
print("\nStep 4: Generating Final Summary Reports")
print("="*80)
summary_data = []
detailed_data = []
for sector in SECTORS:
    if sector in all_results:
        best_model = None
        best_r2 = -float('inf')
        
        for model_name, result in all_results[sector].items():
            detailed_data.append({
                'Sector': sector,
                'Model': model_name,
                'R2': result['metrics']['R2'],
                'RMSE': result['metrics']['RMSE'],
                'MAE': result['metrics']['MAE'],
                'MAPE': result['metrics'].get('MAPE', np.nan),
                'Directional_Accuracy': result['metrics'].get('Directional_Accuracy', np.nan)
            })
            
            if result['metrics']['R2'] > best_r2:
                best_r2 = result['metrics']['R2']
                best_model = model_name
        
        if best_model in all_results[sector]:
            metrics = all_results[sector][best_model]['metrics']
            summary_data.append({
                'Sector': sector,
                'Best_Model': best_model,
                'R2': metrics['R2'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE': metrics.get('MAPE', np.nan),
                'Directional_Accuracy': metrics.get('Directional_Accuracy', np.nan),
                'Forecast_Generated': sector in all_forecasts
            })
if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_path = "/Workspace/Users/akshatp@ida.tcsapps.com/Drafts/final_summary_report.csv"
    summary_df.to_csv(summary_path, index=False)
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_path = "/Workspace/Users/akshatp@ida.tcsapps.com/Drafts/final_detailed_report.csv"
    detailed_df.to_csv(detailed_path, index=False)
    
    print("\n📊 FINAL PERFORMANCE SUMMARY:")
    print("="*120)
    print(f"{'Sector':<20} {'Best Model':<15} {'R²':<10} {'RMSE':<15} {'MAE':<15} {'Dir Acc %':<12} {'Forecast':<10}")
    print("-"*120)
    
    for _, row in summary_df.iterrows():
        forecast_status = "✓" if row['Forecast_Generated'] else "✗"
        print(f"{row['Sector']:<20} {row['Best_Model']:<15} {row['R2']:<10.3f} "
              f"₹{row['RMSE']:<13.2f} ₹{row['MAE']:<13.2f} "
              f"{row['Directional_Accuracy'] if not pd.isna(row['Directional_Accuracy']) else 'N/A':<11.1f}% {forecast_status:<10}")
    
    print("-"*120)
    
    model_counts = summary_df['Best_Model'].value_counts()
    print("\n🏆 BEST MODEL DISTRIBUTION:")
    for model, count in model_counts.items():
        print(f"  {model}: {count} sector(s)")
    
    avg_r2 = summary_df['R2'].mean()
    avg_da = summary_df['Directional_Accuracy'].mean()
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"  Average R²: {avg_r2:.3f} ({(avg_r2*100):.1f}% variance explained)")
    print(f"  Average Directional Accuracy: {avg_da:.1f}%")
    print(f"  Forecasts Generated: {summary_df['Forecast_Generated'].sum()}/{len(summary_df)} sectors")
    
    print(f"\n✅ Summary reports saved:")
    print(f"   {summary_path}")
    print(f"   {detailed_path}")
print("\nStep 5: Saving Model Artifacts")
print("="*80)
import joblib
import json
artifacts_saved = 0
for sector in all_results:
    best_model_name = None
    best_r2 = -float('inf')
    for model_name, result in all_results[sector].items():
        if result['metrics']['R2'] > best_r2:
            best_r2 = result['metrics']['R2']
            best_model_name = model_name
    
    if best_model_name and best_model_name in all_results[sector]:
        try:
            artifact_dir = f"/Workspace/Users/akshatp@ida.tcsapps.com/Drafts/final_artifacts_{sector.replace(' ', '_')}"
            os.makedirs(artifact_dir, exist_ok=True)
            
            result = all_results[sector][best_model_name]
            model = result['model']
            
            if 'LSTM' in best_model_name:
                model.save(f"{artifact_dir}/model.h5")
            else:
                joblib.dump(model, f"{artifact_dir}/model.pkl")
            
            if 'scaler' in result:
                joblib.dump(result['scaler'], f"{artifact_dir}/scaler.pkl")
            
            if 'features' in result:
                with open(f"{artifact_dir}/features.json", 'w') as f:
                    json.dump(result['features'], f)
            
            with open(f"{artifact_dir}/metrics.json", 'w') as f:
                json.dump(result['metrics'], f, indent=2)
            
            artifacts_saved += 1
            print(f"✅ {sector}: Artifacts saved")
            
        except:
            print(f"⚠ {sector}: Could not save artifacts")
print(f"\n🎯 PIPELINE EXECUTION COMPLETE")
print("="*80)
print(f"✅ Processed {len(all_results)} sectors")
print(f"✅ Generated {len(all_forecasts)} forecast files")
print(f"✅ Saved {artifacts_saved} model artifacts")
print(f"✅ Created performance plots for all sectors")
print(f"✅ Generated comprehensive reports")