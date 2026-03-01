"""
NIFTY 50 Quick Monitor
Author: Akshat
Description: Quick one-line summary of forecasting results
"""

import json
from pathlib import Path
from collections import Counter
import sys

def quick_monitor(output_dir="output"):
    """Show quick one-line summary of latest forecast"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print("❌ No output directory found!")
        return
    
    # Find latest run
    run_dirs = sorted([d for d in output_path.iterdir() 
                      if d.is_dir() and d.name not in ["dashboard", "best_model"]])
    
    if not run_dirs:
        print("❌ No forecast runs found!")
        return
    
    latest_run = run_dirs[-1]
    
    # Load metrics
    metrics_path = latest_run / "metrics.json"
    if not metrics_path.exists():
        print(f"❌ No metrics found in {latest_run.name}")
        return
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Find best model
    best_model = None
    best_mae = float('inf')
    
    for model, m in metrics.items():
        if m.get('mae', float('inf')) < best_mae:
            best_mae = m['mae']
            best_model = model
    
    # Count historical best models
    best_models = []
    for run_dir in run_dirs[-10:]:  # Last 10 runs
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                run_metrics = json.load(f)
            
            run_best = min(run_metrics.items(), key=lambda x: x[1].get('mae', float('inf')))
            best_models.append(run_best[0])
    
    # Calculate frequency
    model_counts = Counter(best_models)
    most_common = model_counts.most_common(1)[0] if model_counts else None
    
    # Print summary
    print(f"📈 LATEST: {latest_run.name} | Best: {best_model} | MAE: {best_mae:.1f}")
    
    if most_common:
        freq = (most_common[1] / len(best_models)) * 100
        print(f"📊 HISTORY: {most_common[0]} best {most_common[1]}/{len(best_models)} times ({freq:.0f}%)")
    
    # Show forecast trend if available
    forecast_path = latest_run / "future_forecast.csv"
    if forecast_path.exists():
        import pandas as pd
        df = pd.read_csv(forecast_path)
        if len(df) >= 2:
            start = df['Forecasted_Close'].iloc[0]
            end = df['Forecasted_Close'].iloc[-1]
            change = ((end - start) / start) * 100
            trend = "↗️" if change > 0 else "↘️"
            print(f"🔮 FORECAST: ₹{start:.0f} → ₹{end:.0f} {trend} ({change:+.1f}%)")

if __name__ == "__main__":
    quick_monitor("output")