"""
Test cases for DataLoader module
"""

import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = DataLoader()
    
    @patch('data_loader.load_dataset')
    def test_load_dataset_success(self, mock_load_dataset):
        """Test successful dataset loading"""
        # Mock Hugging Face dataset
        mock_dataset = MagicMock()
        mock_train = MagicMock()
        mock_train.to_pandas.return_value = pd.DataFrame({
            'name': ['Restaurant A', 'Restaurant B'],
            'location': ['City A', 'City B']
        })
        mock_dataset.__getitem__.return_value = mock_train
        mock_dataset.keys.return_value = ['train']
        mock_load_dataset.return_value = mock_dataset
        
        # Load dataset
        df = self.loader.load_dataset()
        
        # Assertions
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertIn('name', df.columns)
        mock_load_dataset.assert_called_once_with("ManikaSaini/zomato-restaurant-recommendation")
    
    @patch('data_loader.load_dataset')
    def test_load_dataset_custom_name(self, mock_load_dataset):
        """Test loading with custom dataset name"""
        custom_loader = DataLoader(dataset_name="custom/dataset")
        mock_dataset = MagicMock()
        mock_train = MagicMock()
        mock_train.to_pandas.return_value = pd.DataFrame({'col': [1, 2]})
        mock_dataset.__getitem__.return_value = mock_train
        mock_dataset.keys.return_value = ['train']
        mock_load_dataset.return_value = mock_dataset
        
        custom_loader.load_dataset()
        mock_load_dataset.assert_called_once_with("custom/dataset")
    
    @patch('data_loader.load_dataset')
    def test_load_dataset_failure(self, mock_load_dataset):
        """Test dataset loading failure"""
        mock_load_dataset.side_effect = Exception("Connection error")
        
        with self.assertRaises(Exception):
            self.loader.load_dataset()
    
    @patch('data_loader.load_dataset')
    def test_get_info_before_loading(self, mock_load_dataset):
        """Test get_info before loading dataset"""
        info = self.loader.get_info()
        self.assertIn("error", info)
    
    @patch('data_loader.load_dataset')
    def test_get_info_after_loading(self, mock_load_dataset):
        """Test get_info after loading dataset"""
        mock_dataset = MagicMock()
        mock_train = MagicMock()
        mock_train.to_pandas.return_value = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        mock_dataset.__getitem__.return_value = mock_train
        mock_dataset.keys.return_value = ['train']
        mock_load_dataset.return_value = mock_dataset
        
        self.loader.load_dataset()
        info = self.loader.get_info()
        
        self.assertIn("shape", info)
        self.assertIn("columns", info)
        self.assertEqual(info["shape"], (3, 2))
        self.assertEqual(len(info["columns"]), 2)
    
    @patch('data_loader.load_dataset')
    def test_get_dataframe(self, mock_load_dataset):
        """Test get_dataframe method"""
        mock_dataset = MagicMock()
        mock_train = MagicMock()
        test_df = pd.DataFrame({'test': [1, 2]})
        mock_train.to_pandas.return_value = test_df
        mock_dataset.__getitem__.return_value = mock_train
        mock_dataset.keys.return_value = ['train']
        mock_load_dataset.return_value = mock_dataset
        
        self.loader.load_dataset()
        df = self.loader.get_dataframe()
        
        self.assertIsNotNone(df)
        pd.testing.assert_frame_equal(df, test_df)


if __name__ == '__main__':
    unittest.main()
