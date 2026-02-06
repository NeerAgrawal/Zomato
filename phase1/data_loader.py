"""
Data Loading Module - STEP 1.1
Purpose: Fetch dataset from Hugging Face
"""

from datasets import load_dataset
import pandas as pd
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load Zomato dataset from Hugging Face"""
    
    def __init__(self, dataset_name: str = "ManikaSaini/zomato-restaurant-recommendation"):
        """
        Initialize DataLoader
        
        Args:
            dataset_name: Hugging Face dataset identifier
        """
        self.dataset_name = dataset_name
        self.raw_dataset = None
        self.df = None
    
    def load_dataset(self) -> pd.DataFrame:
        """
        Load dataset from Hugging Face
        
        Returns:
            pandas.DataFrame: Raw dataset as DataFrame
            
        Raises:
            Exception: If dataset loading fails
        """
        try:
            logger.info(f"Loading dataset: {self.dataset_name}")
            self.raw_dataset = load_dataset(self.dataset_name)
            
            # Convert to pandas DataFrame
            # Usually the dataset has a 'train' split
            if 'train' in self.raw_dataset:
                self.df = self.raw_dataset['train'].to_pandas()
            else:
                # If no train split, use the first available split
                split_name = list(self.raw_dataset.keys())[0]
                self.df = self.raw_dataset[split_name].to_pandas()
            
            logger.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Columns: {list(self.df.columns)}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get the loaded DataFrame
        
        Returns:
            pandas.DataFrame or None: The loaded DataFrame
        """
        return self.df
    
    def get_info(self) -> dict:
        """
        Get basic information about the loaded dataset
        
        Returns:
            dict: Dataset information
        """
        if self.df is None:
            return {"error": "Dataset not loaded"}
        
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
            "dtypes": self.df.dtypes.to_dict()
        }
