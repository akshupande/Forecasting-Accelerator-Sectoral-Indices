
"""
NIFTY AUTO Forecasting
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pywt
import warnings
import joblib
import json
from datetime import datetime, timedelta
import os
import glob
import sys

warnings.filterwarnings('ignore')

# ==============================
# Configuration
# ==============================
# Default data directory (Windows raw string for safety)
DATA_DIR = r"C:\Users\2793478\Downloads\Misc\Lab_Main\Task 3_Nifty Auto_Haar"

# ==============================
# Setup Class
# ==============================
class SmartMLops:
    def __init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = f"run_{self.run_id}"
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"SMART MLops Run: {self.run_id}")
        print(f"Output Directory: {self.run_dir}")
    
    def save_model(self, model, model_name):
        filename = os.path.join(self.run_dir, f"{model_name}.pkl")
        joblib.dump(model, filename)
        print(f"  Saved model: {filename}")
        return filename
    
    def save_metrics(self, metrics, model_name):
        filename = os.path.join(self.run_dir, f"{model_name}_metrics.json")
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved metrics: {filename}")
    
    def save_predictions(self, df, filename="predictions.csv"):
        filepath = os.path.join(self.run_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"  Saved predictions: {filepath}")
        df.to_csv("latest_predictions.csv", index=False)
        return filepath

# ==============================
# Load Data
# ==============================
def load_and_merge_data(data_dir=DATA_DIR, recursive=True):
    print("\nSMART DATA LOADER")
    print("="*50)
    scan_dir = os.path.abspath(data_dir)
    print(f"Scanning data directory: {scan_dir} (recursive={recursive})")

    # Collect CSVs robustly (case-insensitive)
    all_files = []
    for root, _, files in os.walk(scan_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                all_files.append(os.path.join(root, f))
        if not recursive:
            break

    if not all_files:
        print("No CSV files found in the specified data directory.")
        return None

    print(f"Total CSV files found: {len(all_files)}")

    # Exclude obvious non-source files
    exclude_tokens = {'log', 'latest_', 'predictions', 'forecast_', 'summary', 'metrics'}
    candidate_files = []
    for file in all_files:
        filename = os.path.basename(file).lower()
        if any(tok in filename for tok in exclude_tokens):
            continue
        candidate_files.append(file)

    if not candidate_files:
        print("All found CSVs were excluded by exclude filters.")
        return None

    print(f"Found {len(candidate_files)} candidate CSVs:")
    for p in candidate_files:
        print("  -", os.path.relpath(p, start=scan_dir))

    # Whitelist aligned to your naming convention
    nifty_data_files = []
    for file in candidate_files:
        filename = os.path.basename(file).lower()
        if ('nifty' in filename and 'auto' in filename) or ('nifty' in filename and 'historical' in filename):
            nifty_data_files.append(file)
        elif 'historical_pr' in filename:
            nifty_data_files.append(file)

    if not nifty_data_files:
        print("No NIFTY price data found after whitelist filter!")
        return None

    print(f"\nUsing {len(nifty_data_files)} NIFTY data files:")
    for p in nifty_data_files:
        print("  ->", os.path.basename(p))

    # Load, normalize, and merge
    all_data = []
    for file in nifty_data_files:
        try:
            df = pd.read_csv(file)
            # Normalize common column variants to expected names
            rename_map = {}
            for c in df.columns:
                cl = c.strip().lower()
                if cl in ('close price', 'adj close', 'closing price', 'close'):
                    rename_map[c] = 'Close'
                elif cl in ('open price', 'open'):
                    rename_map[c] = 'Open'
                elif cl in ('high price', 'high'):
                    rename_map[c] = 'High'
                elif cl in ('low price', 'low'):
                    rename_map[c] = 'Low'
                elif cl in ('timestamp', 'date time', 'time', 'datetime', 'date'):
                    rename_map[c] = 'Date'
            if rename_map:
                df = df.rename(columns=rename_map)

            print(f"  {os.path.basename(file)} - {len(df)} rows")
            all_data.append(df)
        except Exception as e:
            print(f"  Error reading {file}: {e}")

    if not all_data:
        print("No NIFTY price data could be loaded from the selected files.")
        return None

    merged = pd.concat(all_data, ignore_index=True)
    print(f"\nSuccessfully loaded {len(merged)} rows of NIFTY data")
    return merged

def intelligent_split(data, test_days=60):
    print("\nINTELLIGENT SPLIT")
    print("="*50)
    
    def preprocess_data(df):
        df = df.copy()
        # Date parsing: try multiple formats, including 'DD Mon YYYY', ISO, and 'DDMMYYYY'
        parsed = False
        if 'Date' in df.columns:
            # Strip whitespace
            df['Date'] = df['Date'].astype(str).str.strip()
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%d %b %Y')
                parsed = True
            except Exception:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], errors='raise')
                    parsed = True
                except Exception:
                    try:
                        # Handle numeric-looking dates like 01112025 (DDMMYYYY)
                        df['Date'] = pd.to_datetime(df['Date'], format='%d%m%Y', errors='raise')
                        parsed = True
                    except Exception:
                        parsed = False

        if not parsed:
            # If Date missing or parsing failed, synthesize a daily index
            df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')

        # Clean numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close']
        for col in numeric_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.sort_values('Date')
        if 'Close' in df.columns:
            df = df.dropna(subset=['Close'])
        
        return df.reset_index(drop=True)
    
    data = preprocess_data(data)
    data = data.sort_values('Date').reset_index(drop=True)
    
    split_idx = len(data) - test_days
    if split_idx < int(len(data) * 0.7):
        split_idx = int(len(data) * 0.7)
    
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    print(f"Complete data: {data['Date'].min().date()} to {data['Date'].max().date()}")
    print(f"Total days: {len(data)}")
    print(f"\nTraining: {train['Date'].min().date()} to {train['Date'].max().date()}")
    print(f"Days: {len(train)} ({len(train)/len(data)*100:.1f}%)")
    print(f"\nTesting:  {test['Date'].min().date()} to {test['Date'].max().date()}")
    print(f"Days: {len(test)} ({len(test)/len(data)*100:.1f}%)")
    
    return train, test

def create_features(df):
    df = df.copy()
    
    required_cols = {'Open', 'High', 'Low', 'Close'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"  Missing columns: {missing}")
        return df
    
    df['HL_Ratio'] = df['High'] / df['Low']
    df['OC_Ratio'] = df['Open'] / df['Close']
    df['Returns'] = df['Close'].pct_change()
    df['Returns_1'] = df['Returns'].shift(1)
    df['Returns_2'] = df['Returns'].shift(2)
    
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    
    windows = [5, 10, 20, 50]
    for window in windows:
        df[f'MA_{window}'] = df['Close'].rolling(window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
    
    df['Volatility_5'] = df['Returns'].rolling(5).std()
    df['Volatility_10'] = df['Returns'].rolling(10).std()
    df['Volatility_20'] = df['Returns'].rolling(20).std()
    
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['ATR_5'] = df['TR'].rolling(5).mean()
    df['ATR_10'] = df['TR'].rolling(10).mean()
    
    df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['HL_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['OC_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    return df

# ==============================
# Main Pipeline
# ==============================
def main(data_dir=DATA_DIR):
    print("\n" + "="*70)
    print("NIFTY AUTO FORECASTING - SMART MLOPS")
    print("="*70)
    
    mlops = SmartMLops()
    
    print("\n[1/9] Loading ALL historical data...")
    all_data = load_and_merge_data(data_dir=data_dir, recursive=True)
    
    if all_data is None or len(all_data) == 0:
        print("No data to process!")
        return None
    
    print("\n[2/9] Splitting data intelligently...")
    train, test = intelligent_split(all_data, test_days=60)
    
    print("\n[3/9] Creating advanced features...")
    train_fe = create_features(train)
    test_fe = create_features(test)
    
    print("\n[4/9] Applying wavelet denoising...")
    def wavelet_denoise(data, wavelet='haar', level=1):
        coeff = pywt.wavedec(data, wavelet, level=level)
        sigma = np.median(np.abs(coeff[-level])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(data)))
        coeff[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:]]
        reconstructed = pywt.waverec(coeff, wavelet)
        return reconstructed[:len(data)]
    
    try:
        train_fe['Close_DWT'] = pd.Series(wavelet_denoise(train_fe['Close'].values), index=train_fe.index)
        test_fe['Close_DWT'] = pd.Series(wavelet_denoise(test_fe['Close'].values), index=test_fe.index)
        print("  Wavelet denoising applied")
    except Exception as e:
        print(f"  Wavelet failed: {e}")
        train_fe['Close_DWT'] = train_fe['Close']
        test_fe['Close_DWT'] = test_fe['Close']
    
    train_fe = train_fe.fillna(method='ffill').fillna(method='bfill')
    test_fe = test_fe.fillna(method='ffill').fillna(method='bfill')
    
    print("\n[5/9] Preparing features for ML...")
    exclude_cols = ['Date', 'Close', 'Open', 'High', 'Low', 'Index Name', 'INDEX NAME']
    existing_exclude = [col for col in exclude_cols if col in train_fe.columns]
    feature_cols = [col for col in train_fe.columns if col not in existing_exclude]
    
    print(f"  Features: {len(feature_cols)}")
    
    X_train = train_fe[feature_cols]
    y_train = train_fe['Close']
    X_test = test_fe[feature_cols]
    y_test = test_fe['Close']
    
    print(f"  Training: {X_train.shape}")
    print(f"  Testing:  {X_test.shape}")
    
    print("\n[6/9] Training multiple models...")
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Lasso': Lasso(alpha=0.001, random_state=42, max_iter=10000),
        'Ridge': Ridge(alpha=1.0, random_state=42),
    }
    
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        print("  XGBoost available")
    except Exception:
        print("  XGBoost not available")
        models['XGBoost'] = RandomForestRegressor(n_estimators=100, random_state=42)
    
    results = []
    predictions_data = {'Date': test_fe['Date'].values, 'Actual': y_test.values}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        
        try:
            model.fit(X_train, y_train)
            mlops.save_model(model, name)
            
            y_pred = model.predict(X_test)
            predictions_data[f'Pred_{name}'] = y_pred
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            metrics = {
                'model': name,
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape)
            }
            
            mlops.save_metrics(metrics, name)
            results.append(metrics)
            
            print(f"    MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"    {name} failed: {e}")
    
    if not results:
        print("All models failed!")
        return None
    
    print("\n[7/9] Saving results...")
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
    mlops.save_predictions(predictions_df)
    
    best_model = min(results, key=lambda x: x['rmse'])
    print(f"\nBEST MODEL: {best_model['model']}")
    print(f"   RMSE: {best_model['rmse']:.2f}")
    print(f"   R2:   {best_model['r2']:.4f}")
    print(f"   MAPE: {best_model['mape']:.2f}%")
    
    print("\n[8/9] Saving summary and logs...")
    summary = {
        'run_id': mlops.run_id,
        'timestamp': datetime.now().isoformat(),
        'best_model': best_model['model'],
        'best_rmse': best_model['rmse'],
        'best_r2': best_model['r2'],
        'best_mape': best_model['mape'],
        'train_period': f"{train['Date'].min().date()} to {train['Date'].max().date()}",
        'test_period': f"{test['Date'].min().date()} to {test['Date'].max().date()}",
        'train_days': len(train),
        'test_days': len(test),
        'models_trained': len(results),
        'all_results': results
    }
    
    with open(os.path.join(mlops.run_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    log_entry = pd.DataFrame([{
        'run_id': mlops.run_id,
        'timestamp': summary['timestamp'],
        'best_model': best_model['model'],
        'best_rmse': best_model['rmse'],
        'best_r2': best_model['r2'],
        'test_start': test['Date'].min().date(),
        'test_end': test['Date'].max().date(),
        'test_days': len(test)
    }])
    
    if os.path.exists('smart_mlops_log.csv'):
        existing = pd.read_csv('smart_mlops_log.csv')
        updated = pd.concat([existing, log_entry], ignore_index=True)
    else:
        updated = log_entry
    
    updated.to_csv('smart_mlops_log.csv', index=False)
    
    print("\n" + "="*70)
    print("SMART PIPELINE COMPLETED!")
    print("="*70)
    
    print(f"\nResults saved in: {mlops.run_dir}/")
    print(f"Best Model: {best_model['model']} (RMSE: {best_model['rmse']:.2f})")
    print(f"Test Period: {test['Date'].min().date()} to {test['Date'].max().date()}")
    
    return predictions_df

if __name__ == "__main__":
    print("\n" + "="*70)
    print("NIFTY AUTO SMART MLOPS")
    print("="*70)

    # Optional CLI override: --data-dir "path"
    run_data_dir = DATA_DIR
    if '--data-dir' in sys.argv:
        i = sys.argv.index('--data-dir')
        if i + 1 < len(sys.argv):
            run_data_dir = sys.argv[i + 1]
            # Normalize to absolute path
            run_data_dir = os.path.abspath(run_data_dir)
            print(f"Using data directory from CLI: {run_data_dir}")

    if len(sys.argv) > 1 and sys.argv[1] == '--monitor':
        if os.path.exists('smart_mlops_log.csv'):
            log = pd.read_csv('smart_mlops_log.csv')
            print("\nExperiment History:")
            print(log.to_string(index=False))
            
            if len(log) > 0:
                best_run = log.loc[log['best_rmse'].idxmin()]
                print(f"\nBest Run Overall: {best_run['run_id']}")
                print(f"   Model: {best_run['best_model']}")
                print(f"   RMSE: {best_run['best_rmse']:.2f}")
                print(f"   Tested on: {best_run['test_start']} to {best_run['test_end']}")
    else:
        print("\nRunning pipeline...")
        _ = main(data_dir=run_data_dir)