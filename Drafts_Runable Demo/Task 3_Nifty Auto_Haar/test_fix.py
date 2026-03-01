import pandas as pd
import glob

files = glob.glob("*.csv")
print(f"Total CSV files: {len(files)}")

for f in files:
    try:
        df = pd.read_csv(f, nrows=2)
        print(f"\n{f} ({os.path.getsize(f)/1024:.1f} KB)")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample data: {df.iloc[0].to_dict() if len(df) > 0 else 'Empty'}")
    except:
        print(f"\n{f} - Could not read")