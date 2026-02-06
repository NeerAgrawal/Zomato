"""
Test cases for DataStorage module
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_storage import DataStorage


class TestDataStorage(unittest.TestCase):
    """Test cases for DataStorage"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        self.storage = DataStorage(output_dir=self.test_dir)
        
        # Create sample DataFrame
        self.sample_df = pd.DataFrame({
            'restaurant_id': [1, 2, 3],
            'name': ['Restaurant A', 'Restaurant B', 'Restaurant C'],
            'rate': [4.1, 4.5, 3.8],
            'votes': [100, 200, 150],
            'price': [800, 1000, 600],
            'cuisines': [['North Indian', 'Chinese'], ['Italian'], ['Mexican']],
            'location': ['City A', 'City B', 'City A']
        })
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    def test_save_to_csv(self):
        """Test saving to CSV"""
        filepath = self.storage.save_to_csv(self.sample_df, 'test.csv')
        
        self.assertTrue(os.path.exists(filepath))
        self.assertEqual(os.path.basename(filepath), 'test.csv')
        
        # Verify file can be read back
        loaded_df = pd.read_csv(filepath)
        self.assertEqual(len(loaded_df), 3)
    
    def test_load_from_csv(self):
        """Test loading from CSV"""
        # Save first
        self.storage.save_to_csv(self.sample_df, 'test.csv')
        
        # Load
        loaded_df = self.storage.load_from_csv('test.csv')
        
        self.assertEqual(len(loaded_df), 3)
        self.assertIn('name', loaded_df.columns)
    
    def test_load_from_csv_not_found(self):
        """Test loading from non-existent CSV"""
        with self.assertRaises(FileNotFoundError):
            self.storage.load_from_csv('nonexistent.csv')
    
    def test_save_to_parquet(self):
        """Test saving to Parquet"""
        filepath = self.storage.save_to_parquet(self.sample_df, 'test.parquet')
        
        self.assertTrue(os.path.exists(filepath))
        self.assertEqual(os.path.basename(filepath), 'test.parquet')
    
    def test_load_from_parquet(self):
        """Test loading from Parquet"""
        # Save first
        self.storage.save_to_parquet(self.sample_df, 'test.parquet')
        
        # Load
        loaded_df = self.storage.load_from_parquet('test.parquet')
        
        self.assertEqual(len(loaded_df), 3)
        self.assertIn('name', loaded_df.columns)
    
    def test_load_from_parquet_not_found(self):
        """Test loading from non-existent Parquet"""
        with self.assertRaises(FileNotFoundError):
            self.storage.load_from_parquet('nonexistent.parquet')
    
    def test_save_to_pickle(self):
        """Test saving to Pickle"""
        filepath = self.storage.save_to_pickle(self.sample_df, 'test.pkl')
        
        self.assertTrue(os.path.exists(filepath))
        self.assertEqual(os.path.basename(filepath), 'test.pkl')
    
    def test_load_from_pickle(self):
        """Test loading from Pickle"""
        # Save first
        self.storage.save_to_pickle(self.sample_df, 'test.pkl')
        
        # Load
        loaded_df = self.storage.load_from_pickle('test.pkl')
        
        self.assertEqual(len(loaded_df), 3)
        self.assertIn('name', loaded_df.columns)
        # Pickle preserves list types
        self.assertIsInstance(loaded_df.loc[0, 'cuisines'], list)  # Pickle preserves original types
    
    def test_load_from_pickle_not_found(self):
        """Test loading from non-existent Pickle"""
        with self.assertRaises(FileNotFoundError):
            self.storage.load_from_pickle('nonexistent.pkl')
    
    def test_multiple_formats(self):
        """Test saving in multiple formats"""
        csv_path = self.storage.save_to_csv(self.sample_df, 'multi.csv')
        parquet_path = self.storage.save_to_parquet(self.sample_df, 'multi.parquet')
        pickle_path = self.storage.save_to_pickle(self.sample_df, 'multi.pkl')
        
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(parquet_path))
        self.assertTrue(os.path.exists(pickle_path))
        
        # Verify all can be loaded
        csv_df = self.storage.load_from_csv('multi.csv')
        parquet_df = self.storage.load_from_parquet('multi.parquet')
        pickle_df = self.storage.load_from_pickle('multi.pkl')
        
        self.assertEqual(len(csv_df), 3)
        self.assertEqual(len(parquet_df), 3)
        self.assertEqual(len(pickle_df), 3)


if __name__ == '__main__':
    unittest.main()
