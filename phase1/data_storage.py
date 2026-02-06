"""
Data Storage Module - STEP 1.4
Purpose: Store processed data for efficient access
"""

import pandas as pd
import os
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataStorage:
    """Store processed Zomato dataset"""
    
    def __init__(self, output_dir: str = "data"):
        """
        Initialize DataStorage
        
        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = "processed_zomato_data.csv") -> str:
        """
        Save DataFrame to CSV file
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            str: Path to saved file
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # For columns with lists, convert to string representation
        df_to_save = df.copy()
        for col in df_to_save.columns:
            if df_to_save[col].dtype == 'object':
                # Check if column contains lists
                if df_to_save[col].notna().any():
                    sample = df_to_save[col].dropna().iloc[0]
                    if isinstance(sample, list):
                        df_to_save[col] = df_to_save[col].apply(
                            lambda x: str(x) if x is not None else ''
                        )
        
        df_to_save.to_csv(filepath, index=False)
        logger.info(f"Data saved to CSV: {filepath}")
        return filepath
    
    def save_to_parquet(self, df: pd.DataFrame, filename: str = "processed_zomato_data.parquet") -> str:
        """
        Save DataFrame to Parquet file
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            str: Path to saved file
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # Parquet doesn't support list columns directly, convert to string
        df_to_save = df.copy()
        for col in df_to_save.columns:
            if df_to_save[col].dtype == 'object':
                if df_to_save[col].notna().any():
                    sample = df_to_save[col].dropna().iloc[0]
                    if isinstance(sample, list):
                        df_to_save[col] = df_to_save[col].apply(
                            lambda x: str(x) if x is not None else ''
                        )
        
        df_to_save.to_parquet(filepath, index=False, engine='pyarrow')
        logger.info(f"Data saved to Parquet: {filepath}")
        return filepath
    
    def save_to_pickle(self, df: pd.DataFrame, filename: str = "processed_zomato_data.pkl") -> str:
        """
        Save DataFrame to Pickle file (preserves data types including lists)
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            str: Path to saved file
        """
        filepath = os.path.join(self.output_dir, filename)
        df.to_pickle(filepath)
        logger.info(f"Data saved to Pickle: {filepath}")
        return filepath
    
    def load_from_csv(self, filename: str = "processed_zomato_data.csv") -> pd.DataFrame:
        """
        Load DataFrame from CSV file
        
        Args:
            filename: Input filename
            
        Returns:
            pandas.DataFrame: Loaded DataFrame
        """
        filepath = os.path.join(self.output_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded from CSV: {filepath}")
        return df
    
    def load_from_parquet(self, filename: str = "processed_zomato_data.parquet") -> pd.DataFrame:
        """
        Load DataFrame from Parquet file
        
        Args:
            filename: Input filename
            
        Returns:
            pandas.DataFrame: Loaded DataFrame
        """
        filepath = os.path.join(self.output_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_parquet(filepath, engine='pyarrow')
        logger.info(f"Data loaded from Parquet: {filepath}")
        return df
    
    def load_from_pickle(self, filename: str = "processed_zomato_data.pkl") -> pd.DataFrame:
        """
        Load DataFrame from Pickle file
        
        Args:
            filename: Input filename
            
        Returns:
            pandas.DataFrame: Loaded DataFrame
        """
        filepath = os.path.join(self.output_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_pickle(filepath)
        logger.info(f"Data loaded from Pickle: {filepath}")
        return df
