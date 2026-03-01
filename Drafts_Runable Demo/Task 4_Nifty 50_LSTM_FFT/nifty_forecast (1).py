"""
NIFTY 50 Forecasting Production Script
Author: Akshat
"""

import os
import sys
import argparse
import logging
import warnings
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


warnings.filterwarnings("ignore")


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
import random
random.seed(SEED)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nifty_forecasting.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NiftyForecaster:
    
    def __init__(self, data_dir: str = ".", output_dir: str = "output", 
                 seq_window: int = 30, fft_topk: int = 5):

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.seq_window = seq_window
        self.fft_topk = fft_topk
        
        # Data storage
        self.df = None
        self.train_df = None
        self.test_df = None
        self.features_with_fft = None
        self.features_without_fft = None
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.predictions = {}
        
        # Timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Initialized NiftyForecaster with data_dir={data_dir}, output_dir={output_dir}")
        
    def find_csv_files(self) -> List[Path]:
        csv_files = list(self.data_dir.glob("*.csv"))
        csv_files = [f for f in csv_files if "NIFTY" in f.name.upper()]
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
            
        logger.info(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
        return csv_files
    
    def load_and_preprocess(self, csv_files: List[Path]) -> pd.DataFrame:
        
        logger.info("Loading and preprocessing data...")
        
 
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df.columns = df.columns.str.strip()
                dfs.append(df)
                logger.info(f"Loaded {file.name} with shape {df.shape}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No data could be loaded from CSV files")
        
        
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined data shape: {df.shape}")
        
        
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date').dropna(subset=['Date', 'Close']).reset_index(drop=True)
        
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Shares Traded', 'Turnover (₹ Cr)']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Performing feature engineering...")
        
        df = df.copy()
        
        # Moving averages and exponential moving averages
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA30'] = df['Close'].rolling(30).mean()
        df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA30'] = df['Close'].ewm(span=30, adjust=False).mean()
        
        # Daily changes and returns
        df['Daily_Change'] = df['Close'].diff()
        df['Returns'] = df['Close'].pct_change()
        df['Volatility10'] = df['Returns'].rolling(10).std()
        df['Volatility20'] = df['Returns'].rolling(20).std()
        df['High_Low_Range'] = df['High'] - df['Low']
        
        # Lag features
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
        
        # Bollinger Bands
        ma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        df['BB_Upper'] = ma20 + 2 * std20
        df['BB_Lower'] = ma20 - 2 * std20
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        
        # FFT features
        logger.info("Computing FFT features...")
        df = self._compute_fft_features(df)
        
        return df
    
    def _compute_fft_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        
        def compute_fft_features_window(arr, top_k=5):
            
            x = arr - np.mean(arr)
            fft_vals = np.fft.fft(x)
            fft_mag = np.abs(fft_vals)[:len(fft_vals)//2]
            
            if fft_mag.sum() == 0:
                p = np.ones_like(fft_mag) / len(fft_mag)
            else:
                p = fft_mag / (fft_mag.sum() + 1e-12)
            
            dom_idx = np.argmax(fft_mag[1:]) + 1 if len(fft_mag) > 1 else 0
            dom_freq_norm = dom_idx / len(fft_mag) if len(fft_mag) > 0 else 0
            spectral_energy = np.sum(fft_mag**2)
            spectral_entropy = -np.sum(p * np.log(p + 1e-12))
            topk_avg = np.mean(np.sort(fft_mag)[-top_k:]) if len(fft_mag) >= top_k else np.mean(fft_mag)
            
            return dom_freq_norm, spectral_energy, spectral_entropy, topk_avg
        
        
        df['FFT_DominantFreq'] = np.nan
        df['FFT_Energy'] = np.nan
        df['FFT_Entropy'] = np.nan
        df['FFT_TopKAvg'] = np.nan
        
        
        close_vals = df['Close'].values
        for i in range(self.seq_window - 1, len(df)):
            window_vals = close_vals[i - self.seq_window + 1: i + 1]
            dom, energy, entropy, topk = compute_fft_features_window(window_vals, top_k=self.fft_topk)
            df.at[i, 'FFT_DominantFreq'] = dom
            df.at[i, 'FFT_Energy'] = energy
            df.at[i, 'FFT_Entropy'] = entropy
            df.at[i, 'FFT_TopKAvg'] = topk
        
        return df.dropna().reset_index(drop=True)
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        logger.info(f"Splitting data with train ratio: {train_ratio}")
        
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        
        turnover_max = train_df['Turnover (₹ Cr)'].max()
        volume_max = train_df['Shares Traded'].max()
        
        train_df['Turnover_Norm'] = train_df['Turnover (₹ Cr)'] / (turnover_max + 1e-12)
        train_df['Volume_Norm'] = train_df['Shares Traded'] / (volume_max + 1e-12)
        
        test_df['Turnover_Norm'] = test_df['Turnover (₹ Cr)'] / (turnover_max + 1e-12)
        test_df['Volume_Norm'] = test_df['Shares Traded'] / (volume_max + 1e-12)
        
        logger.info(f"Train data: {len(train_df)} rows from {train_df['Date'].min()} to {train_df['Date'].max()}")
        logger.info(f"Test data: {len(test_df)} rows from {test_df['Date'].min()} to {test_df['Date'].max()}")
        
        return train_df, test_df
    
    def create_sequences(self, data: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(data[i, 0])  # First column is Close price
        return np.array(X), np.array(y)
    
    def create_flat_sequences(self, data: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i].reshape(-1))
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def build_lstm(self, input_shape: Tuple, bidirectional: bool = True) -> Sequential:
        
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
    
    def train_lstm(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                   features: List[str], model_name: str, 
                   bidirectional: bool = True, epochs: int = 50) -> Dict[str, Any]:
        
        logger.info(f"Training LSTM model: {model_name}")
        
        # Prepare data
        scaler = MinMaxScaler()
        train_feats = train_df[features].values
        test_feats = test_df[features].values
        
        scaler.fit(train_feats)
        train_scaled = scaler.transform(train_feats)
        test_scaled = scaler.transform(test_feats)
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_scaled, self.seq_window)
        X_test, y_test = self.create_sequences(test_scaled, self.seq_window)
        
        # Build and train model
        model = self.build_lstm((X_train.shape[1], X_train.shape[2]), bidirectional)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            verbose=0,
            validation_split=0.1,
            callbacks=callbacks
        )
        
        
        pred_test_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
        
        
        pad_pred = np.hstack([pred_test_scaled, np.zeros((pred_test_scaled.shape[0], len(features)-1))])
        pred_test_rescaled = scaler.inverse_transform(pad_pred)[:, 0]
        
        pad_actual = np.hstack([y_test.reshape(-1,1), np.zeros((y_test.shape[0], len(features)-1))])
        actual_test_rescaled = scaler.inverse_transform(pad_actual)[:, 0]
        
        
        metrics = self.calculate_metrics(actual_test_rescaled, pred_test_rescaled)
        
        
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.metrics[model_name] = metrics
        
        
        dates = test_df['Date'].values[self.seq_window:]
        self.predictions[model_name] = {
            'dates': dates,
            'actual': actual_test_rescaled,
            'predicted': pred_test_rescaled,
            'features': features  
        }
        
        logger.info(f"LSTM {model_name} trained: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
        
        return metrics
    
    def train_ml_models(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                        features: List[str], model_name_prefix: str) -> Dict[str, Dict]:
        
        logger.info(f"Training ML models: {model_name_prefix}")
        
        # Prepare data
        scaler = MinMaxScaler()
        train_feats = train_df[features].values
        test_feats = test_df[features].values
        
        scaler.fit(train_feats)
        train_scaled = scaler.transform(train_feats)
        test_scaled = scaler.transform(test_feats)
        
        
        X_train, y_train = self.create_flat_sequences(train_scaled, self.seq_window)
        X_test, y_test = self.create_flat_sequences(test_scaled, self.seq_window)
        
        
        models_dict = {
            'RandomForest': RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=200, random_state=SEED, n_jobs=-1, objective='reg:squarederror'),
            'Lasso': Lasso(alpha=0.001, random_state=SEED, max_iter=5000),
            'Ridge': Ridge(alpha=1.0, random_state=SEED)
        }
        
        results = {}
        for name, model in models_dict.items():
            full_name = f"{model_name_prefix}_{name}"
            logger.info(f"Training {full_name}...")
            
            
            model.fit(X_train, y_train)
            
            
            pred_test_scaled = model.predict(X_test).reshape(-1, 1)
            
            
            pad_pred = np.hstack([pred_test_scaled, np.zeros((pred_test_scaled.shape[0], len(features)-1))])
            pred_test_rescaled = scaler.inverse_transform(pad_pred)[:, 0]
            
            pad_actual = np.hstack([y_test.reshape(-1,1), np.zeros((y_test.shape[0], len(features)-1))])
            actual_test_rescaled = scaler.inverse_transform(pad_actual)[:, 0]
            
            
            metrics = self.calculate_metrics(actual_test_rescaled, pred_test_rescaled)
            
            
            self.models[full_name] = model
            self.scalers[full_name] = scaler
            self.metrics[full_name] = metrics
            
            
            dates = test_df['Date'].values[self.seq_window:]
            self.predictions[full_name] = {
                'dates': dates,
                'actual': actual_test_rescaled,
                'predicted': pred_test_rescaled,
                'features': features  
            }
            
            results[name] = metrics
            logger.info(f"{full_name}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
        
        return results
    
    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-12))) * 100
        
        
        if len(actual) > 1 and len(predicted) > 1:
            actual_changes = np.diff(actual)
            predicted_changes = np.diff(predicted)
            directional_acc = np.mean(np.sign(actual_changes) == np.sign(predicted_changes))
        else:
            directional_acc = 0.0
        
        # R² score
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-12))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_acc,
            'r2_score': r2
        }
    
    def prepare_data_for_forecast(self, features: List[str]) -> pd.DataFrame:

        forecast_df = self.df.copy()

        if self.train_df is not None:
            turnover_max = self.train_df['Turnover (₹ Cr)'].max()
            volume_max = self.train_df['Shares Traded'].max()
            
            forecast_df['Turnover_Norm'] = forecast_df['Turnover (₹ Cr)'] / (turnover_max + 1e-12)
            forecast_df['Volume_Norm'] = forecast_df['Shares Traded'] / (volume_max + 1e-12)
        else:
      
            turnover_max = forecast_df['Turnover (₹ Cr)'].max()
            volume_max = forecast_df['Shares Traded'].max()
            
            forecast_df['Turnover_Norm'] = forecast_df['Turnover (₹ Cr)'] / (turnover_max + 1e-12)
            forecast_df['Volume_Norm'] = forecast_df['Shares Traded'] / (volume_max + 1e-12)
        
        
        missing_features = [f for f in features if f not in forecast_df.columns]
        if missing_features:
            logger.warning(f"Missing features for forecast: {missing_features}")
            
            for feature in missing_features:
                forecast_df[feature] = 0
        
        return forecast_df
    
    def forecast_future(self, model_name: str, features: List[str], n_days: int = 30) -> pd.DataFrame:
    
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        
        forecast_df = self.prepare_data_for_forecast(features)
        
        
        last_data = forecast_df[features].values[-self.seq_window:]
        last_scaled = scaler.transform(last_data)
        
        forecasts = []
        current_sequence = last_scaled.copy()
        
        for day in range(n_days):
            if isinstance(model, tf.keras.Model):
               
                X_pred = current_sequence.reshape(1, self.seq_window, len(features))
                pred_scaled = model.predict(X_pred, verbose=0)[0, 0]
            else:
                
                X_pred = current_sequence.reshape(1, -1)
                pred_scaled = model.predict(X_pred)[0]
            
            forecasts.append(pred_scaled)
            
          
            new_row = current_sequence[-1].copy()
            new_row[0] = pred_scaled  
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        
        forecast_scaled = np.array(forecasts).reshape(-1, 1)
        pad_forecast = np.hstack([forecast_scaled, np.zeros((len(forecasts), len(features)-1))])
        forecast_rescaled = scaler.inverse_transform(pad_forecast)[:, 0]
        
        
        last_date = forecast_df['Date'].iloc[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]
        
        forecast_df_result = pd.DataFrame({
            'Date': forecast_dates,
            'Forecasted_Close': forecast_rescaled
        })
        
        
        forecast_df_result['Daily_Change'] = forecast_df_result['Forecasted_Close'].diff()
        forecast_df_result['Percent_Change'] = forecast_df_result['Forecasted_Close'].pct_change() * 100
        
        logger.info(f"Generated {n_days}-day forecast using {model_name}")
        
        return forecast_df_result
    
    def save_artifacts(self):
        logger.info("Saving artifacts...")
        
        # Create run directory
        run_dir = self.output_dir / self.run_timestamp
        run_dir.mkdir(exist_ok=True)
        
        # Save models
        models_dir = run_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            # Save Keras models
            if isinstance(model, tf.keras.Model):
                model_path = models_dir / f"{name}.h5"
                model.save(model_path)
            # Save sklearn models
            else:
                model_path = models_dir / f"{name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save scaler
            if name in self.scalers:
                scaler_path = models_dir / f"{name}_scaler.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[name], f)
        
        # Save metrics
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json_metrics = {}
            for model_name, model_metrics in self.metrics.items():
                json_metrics[model_name] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in model_metrics.items()
                }
            json.dump(json_metrics, f, indent=2)
        
        # Save predictions
        predictions_path = run_dir / "predictions.pkl"
        with open(predictions_path, 'wb') as f:
            pickle.dump(self.predictions, f)
        
        # Save feature lists
        features_data = {
            'features_with_fft': self.features_with_fft,
            'features_without_fft': self.features_without_fft
        }
        features_path = run_dir / "features.json"
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump(features_data, f, indent=2)
        
        # Save summary report
        self.save_summary_report(run_dir)
        
        logger.info(f"Artifacts saved to: {run_dir}")
        
        return run_dir
    
    def save_summary_report(self, run_dir: Path):
        report_path = run_dir / "summary_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("NIFTY 50 FORECASTING SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Run Timestamp: {self.run_timestamp}\n")
            f.write(f"Sequence Window: {self.seq_window}\n")
            f.write(f"FFT TopK: {self.fft_topk}\n")
            f.write(f"Train Rows: {len(self.train_df) if self.train_df is not None else 'N/A'}\n")
            f.write(f"Test Rows: {len(self.test_df) if self.test_df is not None else 'N/A'}\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("MODEL PERFORMANCE METRICS\n")
            f.write("=" * 80 + "\n\n")
            
            sorted_models = sorted(self.metrics.items(), key=lambda x: x[1]['mae'])
            
            for model_name, metrics in sorted_models:
                f.write(f"Model: {model_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"MAE: {metrics['mae']:.2f}\n")
                f.write(f"RMSE: {metrics['rmse']:.2f}\n")
                f.write(f"MAPE: {metrics['mape']:.2f}%\n")
                f.write(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}\n")
                f.write(f"R² Score: {metrics['r2_score']:.4f}\n")
                f.write("\n")
            
            # Find best model
            if sorted_models:
                best_model, best_metrics = sorted_models[0]
                f.write("=" * 80 + "\n")
                f.write("BEST MODEL\n")
                f.write("=" * 80 + "\n")
                f.write(f"Model: {best_model}\n")
                f.write(f"MAE: {best_metrics['mae']:.2f}\n")
                f.write(f"RMSE: {best_metrics['rmse']:.2f}\n")
                f.write(f"MAPE: {best_metrics['mape']:.2f}%\n")
                f.write(f"Directional Accuracy: {best_metrics['directional_accuracy']:.2%}\n")
                f.write(f"R² Score: {best_metrics['r2_score']:.4f}\n")
        
        if sorted_models:
            best_model_name = sorted_models[0][0]
            best_model_path = self.output_dir / "best_model"
            best_model_path.mkdir(exist_ok=True)
            
            # Copy best model files
            import shutil
            models_dir = run_dir / "models"
            for file in models_dir.glob(f"{best_model_name}*"):
                shutil.copy(file, best_model_path / file.name)
            
            # Save best model info
            best_model_info = {
                'model_name': best_model_name,
                'timestamp': self.run_timestamp,
                'metrics': self.metrics[best_model_name]
            }
            info_path = best_model_path / "best_model_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(best_model_info, f, indent=2, default=str)
            
            logger.info(f"Best model '{best_model_name}' saved to: {best_model_path}")
    
    def create_plots(self, run_dir: Path):
        """Create and save forecast plots"""
        logger.info("Creating forecast plots...")
        
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        for model_name, pred_data in self.predictions.items():
            plt.figure(figsize=(12, 6))
            plt.plot(pred_data['dates'], pred_data['actual'], label='Actual', linewidth=2)
            plt.plot(pred_data['dates'], pred_data['predicted'], label='Predicted', linewidth=1.5, linestyle='--')
            
            plt.title(f'NIFTY 50 Forecasting - {model_name}')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plot_path = plots_dir / f"{model_name}_forecast.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        # Create comparison plot for best models
        self.create_comparison_plot(run_dir, plots_dir)
        
        logger.info(f"Plots saved to: {plots_dir}")
    
    def create_comparison_plot(self, run_dir: Path, plots_dir: Path):
        """Create comparison plot of top models"""
        if len(self.metrics) < 2:
            return
        
        # Get top 3 models by MAE
        top_models = sorted(self.metrics.items(), key=lambda x: x[1]['mae'])[:3]
        
        plt.figure(figsize=(14, 8))
        
        # Plot actual
        first_model = top_models[0][0]
        actual_dates = self.predictions[first_model]['dates']
        actual_values = self.predictions[first_model]['actual']
        plt.plot(actual_dates, actual_values, label='Actual', linewidth=3, color='black')
        
        # Plot predictions
        colors = ['red', 'blue', 'green']
        for (model_name, _), color in zip(top_models, colors):
            pred_data = self.predictions[model_name]
            plt.plot(pred_data['dates'], pred_data['predicted'], 
                    label=f'{model_name}', linewidth=2, linestyle='--', alpha=0.8)
        
        plt.title('NIFTY 50 Forecasting - Top Model Comparison', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Close Price', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_path = plots_dir / "top_models_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_pipeline(self, train_ratio: float = 0.7, lstm_epochs: int = 50):
        logger.info("Starting NIFTY 50 forecasting pipeline...")
        
        try:
            # 1. Find and load data
            csv_files = self.find_csv_files()
            self.df = self.load_and_preprocess(csv_files)
            
            # 2. Feature engineering
            self.df = self.feature_engineering(self.df)
            
            # 3. Split data
            self.train_df, self.test_df = self.split_data(self.df, train_ratio)
            
            # Define feature sets
            self.features_with_fft = ['Close', 'FFT_DominantFreq', 'FFT_Energy', 'FFT_Entropy', 'FFT_TopKAvg',
                                     'MA10', 'MA30', 'EMA10', 'EMA30', 'Daily_Change', 'Returns',
                                     'Volatility10', 'Volatility20', 'High_Low_Range', 'Turnover_Norm', 'Volume_Norm',
                                     'Close_Lag1', 'Close_Lag2', 'Returns_Lag1', 'Returns_Lag2',
                                     'RSI', 'MACD', 'MACD_Signal', 'BB_Width']
            
            self.features_without_fft = [f for f in self.features_with_fft if not f.startswith('FFT_')]
            
            
            logger.info("Training LSTM models...")
            lstm_fft_metrics = self.train_lstm(self.train_df, self.test_df, self.features_with_fft, 
                                              "LSTM_with_FFT", bidirectional=True, epochs=lstm_epochs)
            
            lstm_nofft_metrics = self.train_lstm(self.train_df, self.test_df, self.features_without_fft, 
                                                "LSTM_without_FFT", bidirectional=True, epochs=lstm_epochs)
            
            
            logger.info("Training ML models...")
            ml_fft_results = self.train_ml_models(self.train_df, self.test_df, self.features_with_fft, "ML_with_FFT")
            ml_nofft_results = self.train_ml_models(self.train_df, self.test_df, self.features_without_fft, "ML_without_FFT")
            
            
            best_model = min(self.metrics.items(), key=lambda x: x[1]['mae'])[0]
            logger.info(f"Generating future forecasts with best model: {best_model}")
            
            
            if "with_FFT" in best_model:
                forecast_features = self.features_with_fft
            else:
                forecast_features = self.features_without_fft
            
            future_forecast = self.forecast_future(best_model, forecast_features, n_days=30)
            
            
            run_dir = self.save_artifacts()
            
            
            forecast_path = run_dir / "future_forecast.csv"
            future_forecast.to_csv(forecast_path, index=False)
            
            
            self.create_plots(run_dir)
            
            
            self.print_summary()
            
            logger.info("Pipeline completed successfully!")
            logger.info(f"Best model: {best_model}")
            logger.info(f"All artifacts saved to: {run_dir}")
            
            return {
                'success': True,
                'run_dir': str(run_dir),
                'best_model': best_model,
                'metrics': self.metrics[best_model],
                'future_forecast_path': str(forecast_path)
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def print_summary(self):
        print("\n" + "=" * 80)
        print("NIFTY 50 FORECASTING - SUMMARY")
        print("=" * 80)
        
        # Sort models by MAE
        sorted_models = sorted(self.metrics.items(), key=lambda x: x[1]['mae'])
        
        print("\nMODEL PERFORMANCE (sorted by MAE):")
        print("-" * 80)
        print(f"{'Model':<30} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'DirAcc':<10}")
        print("-" * 80)
        
        for model_name, metrics in sorted_models:
            print(f"{model_name:<30} {metrics['mae']:<10.2f} {metrics['rmse']:<10.2f} "
                  f"{metrics['mape']:<10.2f} {metrics['directional_accuracy']:<10.2%}")
        
        print("-" * 80)
        
        if sorted_models:
            best_model, best_metrics = sorted_models[0]
            print(f"\nBEST MODEL: {best_model}")
            print(f"MAE: {best_metrics['mae']:.2f}")
            print(f"RMSE: {best_metrics['rmse']:.2f}")
            print(f"MAPE: {best_metrics['mape']:.2f}%")
            print(f"Directional Accuracy: {best_metrics['directional_accuracy']:.2%}")
        
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='NIFTY 50 Forecasting Pipeline')
    
    parser.add_argument('--data-dir', type=str, default='.',
                       help='Directory containing CSV files (default: current directory)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for artifacts (default: output)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Train/test split ratio (default: 0.7)')
    parser.add_argument('--seq-window', type=int, default=30,
                       help='Sequence window length for LSTM (default: 30)')
    parser.add_argument('--fft-topk', type=int, default=5,
                       help='Number of top FFT frequencies (default: 5)')
    parser.add_argument('--lstm-epochs', type=int, default=50,
                       help='Number of epochs for LSTM training (default: 50)')
    parser.add_argument('--forecast-days', type=int, default=30,
                       help='Number of days to forecast (default: 30)')
    
    args = parser.parse_args() #for databricks notebook use -> args, _ = parser.parse_known_args()
    
    try:
        forecaster = NiftyForecaster(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            seq_window=args.seq_window,
            fft_topk=args.fft_topk
        )
        
        result = forecaster.run_pipeline(
            train_ratio=args.train_ratio,
            lstm_epochs=args.lstm_epochs
        )
        
        if result['success']:
            print(f"\n[SUCCESS] Pipeline completed successfully!")
            print(f"[FOLDER] Output directory: {result['run_dir']}")
            print(f"[BEST MODEL] {result['best_model']}")
            print(f"[FORECAST] Future forecast saved to: {result['future_forecast_path']}")
            
            
            dashboard_path = create_simple_dashboard(result, forecaster)
            print(f"[DASHBOARD] Dashboard available at: {dashboard_path}")
            
            
            print("\n" + "=" * 60)
            print("=" * 60)
            print("=" * 60)
            
            sys.exit(0)
        else:
            print(f"\n[ERROR] Pipeline failed: {result['error']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"\n[FATAL ERROR] {str(e)}")
        sys.exit(1)


def create_simple_dashboard(result: Dict, forecaster: NiftyForecaster) -> str:
    dashboard_dir = Path(forecaster.output_dir) / "dashboard"
    dashboard_dir.mkdir(exist_ok=True)
    
    dashboard_path = dashboard_dir / "index.html"
    
    
    forecast_path = Path(result['future_forecast_path'])
    forecast_df = pd.read_csv(forecast_path) if forecast_path.exists() else None
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NIFTY 50 Forecasting Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            .summary {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .metric-card {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ color: #7f8c8d; font-size: 14px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .update-time {{ color: #7f8c8d; font-size: 14px; margin-top: 20px; }}
            .plot {{ margin: 20px 0; text-align: center; }}
            img {{ max-width: 100%; height: auto; border-radius: 5px; }}
            .positive {{ color: green; font-weight: bold; }}
            .negative {{ color: red; font-weight: bold; }}
            .icons {{ font-size: 20px; margin-right: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1><span class="icons">📈</span> NIFTY 50 Forecasting Dashboard</h1>
            
            <div class="summary">
                <h2>Run Summary</h2>
                <p><strong>Timestamp:</strong> {forecaster.run_timestamp}</p>
                <p><strong>Best Model:</strong> {result['best_model']}</p>
                <p><strong>Output Directory:</strong> {result['run_dir']}</p>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{result['metrics']['mae']:.2f}</div>
                    <div class="metric-label">MAE</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{result['metrics']['rmse']:.2f}</div>
                    <div class="metric-label">RMSE</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{result['metrics']['mape']:.2f}%</div>
                    <div class="metric-label">MAPE</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{result['metrics']['directional_accuracy']:.2%}</div>
                    <div class="metric-label">Directional Accuracy</div>
                </div>
            </div>
    """
    
    if forecast_df is not None:
        html_content += """
            <h2><span class="icons">📅</span> 30-Day Forecast</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Forecasted Close</th>
                        <th>Daily Change</th>
                        <th>% Change</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for i, row in forecast_df.iterrows():
            change_class = "positive" if row.get('Daily_Change', 0) >= 0 else "negative"
            pct_class = "positive" if row.get('Percent_Change', 0) >= 0 else "negative"
            
            html_content += f"""
                    <tr>
                        <td>{row['Date']}</td>
                        <td>₹{row['Forecasted_Close']:.2f}</td>
                        <td class="{change_class}">{row.get('Daily_Change', 0):.2f}</td>
                        <td class="{pct_class}">{row.get('Percent_Change', 0):.2f}%</td>
                    </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        """
    
    run_dir = Path(result['run_dir'])
    plots_dir = run_dir / "plots"
    comparison_plot = plots_dir / "top_models_comparison.png"
    
    if comparison_plot.exists():
        rel_plot_path = comparison_plot.relative_to(forecaster.output_dir)
        html_content += f"""
            <div class="plot">
                <h2><span class="icons">📊</span> Model Comparison</h2>
                <img src="../{rel_plot_path}" alt="Model Comparison Plot">
            </div>
        """
    
    html_content += f"""
            <div class="update-time">
                Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            
            <script>
                // Auto-refresh every 5 minutes
                setTimeout(function() {{
                    window.location.reload();
                }}, 300000);
            </script>
        </div>
    </body>
    </html>
    """
    
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    text_dashboard = dashboard_dir / "forecast_summary.txt"
    with open(text_dashboard, 'w', encoding='utf-8') as f:
        f.write("NIFTY 50 FORECAST SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best Model: {result['best_model']}\n")
        f.write(f"MAE: {result['metrics']['mae']:.2f}\n")
        f.write(f"RMSE: {result['metrics']['rmse']:.2f}\n")
        f.write(f"MAPE: {result['metrics']['mape']:.2f}%\n")
        f.write(f"Directional Accuracy: {result['metrics']['directional_accuracy']:.2%}\n\n")
        
        if forecast_df is not None:
            f.write("30-DAY FORECAST:\n")
            f.write("-" * 50 + "\n")
            for _, row in forecast_df.iterrows():
                f.write(f"{row['Date']}: ₹{row['Forecasted_Close']:.2f} ")
                if 'Percent_Change' in row and not pd.isna(row['Percent_Change']):
                    change_sign = "+" if row['Percent_Change'] >= 0 else ""
                    f.write(f"({change_sign}{row['Percent_Change']:.2f}%)\n")
                else:
                    f.write("\n")
    
    return str(dashboard_path)


if __name__ == "__main__":
    main()