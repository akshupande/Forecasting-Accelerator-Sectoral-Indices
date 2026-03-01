# Databricks notebook source
# 1. Install necessary libraries
%pip install git+https://github.com/amazon-science/chronos-forecasting.git matplotlib pandas torch numpy tabulate

import os
import glob
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
from tabulate import tabulate

# --- Configuration ---
BASE_PATH = "/Workspace/Users/akshatp@ida.tcsapps.com/Drafts"
CSV_DIR = f"{BASE_PATH}/csv"
RESULTS_DIR = f"{BASE_PATH}/Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# List of 7 sectoral indices
indices = ["NIFTY 50", "NIFTY AUTO", "NIFTY BANK", "NIFTY ENERGY", "NIFTY IT", "NIFTY MEDIA", "NIFTY METAL"]

# 2. Initialize Chronos Pipeline
# Switched to 'small' to improve accuracy on volatile sectors like Bank/Auto
print("Loading Amazon Chronos-T5-Small model...")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="auto", 
    torch_dtype=torch.bfloat16,
)

def run_complete_workflow():
    all_metrics = []

    for index in indices:
        print(f"\n--- Processing {index} ---")
        
        # Identify files for this specific index
        files = sorted(glob.glob(f"{CSV_DIR}/{index}*.csv"))
        if len(files) < 3:
            print(f"⚠️ Skipping {index}: Found {len(files)} files, need 3.")
            continue

        # Logic: First two files for training, last for testing
        train_paths = files[:2]
        test_path = files[2]

        def load_and_clean(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip() # Remove spaces from headers
            df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%d-%b-%Y')
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            return df.sort_values('Date')

        # Load Data
        df_train = pd.concat([load_and_clean(p) for p in train_paths]).reset_index(drop=True)
        df_test = load_and_clean(test_path)

        # 3. Forecasting with Chronos
        context = torch.tensor(df_train['Close'].values)
        prediction_length = len(df_test)
        
        # Increase num_samples for better probabilistic coverage
        forecast = pipeline.predict(context, prediction_length, num_samples=20)
        
        # Extract median and confidence intervals
        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        df_test['Chronos_Predicted'] = median

        # 4. Calculate Metrics
        mape = np.mean(np.abs((df_test['Close'] - df_test['Chronos_Predicted']) / df_test['Close'])) * 100
        rmse = np.sqrt(np.mean((df_test['Close'] - df_test['Chronos_Predicted'])**2))
        
        # Determine status
        status = "🟢 Excellent" if mape < 5 else "🟡 Good" if mape < 10 else "🔴 Poor"
        all_metrics.append([index, f"{mape:.2f}%", f"{rmse:.2f}", status])

        # 5. Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(df_train['Date'].tail(60), df_train['Close'].tail(60), label="History (Last 60d)", color='gray', alpha=0.6)
        plt.plot(df_test['Date'], df_test['Close'], label="Actual", color='blue', linewidth=2)
        plt.plot(df_test['Date'], df_test['Chronos_Predicted'], label="Chronos Forecast", color='red', linestyle='--')
        plt.fill_between(df_test['Date'], low, high, color='red', alpha=0.1, label="80% Confidence Interval")
        
        plt.title(f"{index} Forecast | MAPE: {mape:.2f}%")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save Outputs
        safe_name = index.replace(" ", "_")
        plt.savefig(f"{RESULTS_DIR}/{safe_name}_forecast.png")
        df_test.to_csv(f"{RESULTS_DIR}/{safe_name}_results.csv", index=False)
        plt.close()

    # 6. Final Terminal Summary
    print("\n" + "="*70)
    print("FINAL CHRONOS FORECASTING SUMMARY")
    print("="*70)
    print(tabulate(all_metrics, headers=["Index", "MAPE", "RMSE", "Status"], tablefmt="grid"))
    print("\nResults and plots saved to:", RESULTS_DIR)

# Run the full process
run_complete_workflow()