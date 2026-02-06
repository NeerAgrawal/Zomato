import pandas as pd
from pathlib import Path

def optimize_data():
    base_path = Path('phase1/data')
    input_file = base_path / 'processed_zomato_data.pkl'
    output_file = base_path / 'zomato_lite.pkl'
    
    print(f"Loading large dataset from {input_file}...")
    if not input_file.exists():
        input_file = base_path / 'processed_zomato_data.csv'
        df = pd.read_csv(input_file)
    else:
        df = pd.read_pickle(input_file)
        
    print(f"Original shape: {df.shape}")
    
    # optimization Strategy:
    # 1. Deduplicate
    df = df.drop_duplicates(subset=['name', 'location'])
    
    # 2. Filter for quality (e.g., has votes and rating)
    # Ensure numeric columns
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0)
    
    # Sort by votes to keep popular restaurants (better for generic recommendation demo)
    df_lite = df.sort_values(by='votes', ascending=False).head(8000)
    
    print(f"Optimized shape: {df_lite.shape}")
    print(f"Saving to {output_file}...")
    
    df_lite.to_pickle(output_file)
    print("Done! This file should be small enough for GitHub (<50MB).")

if __name__ == "__main__":
    optimize_data()
