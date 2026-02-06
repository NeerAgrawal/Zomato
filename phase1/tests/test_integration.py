"""
Integration tests for Phase 1 - STEP 1
Tests the complete pipeline: Load -> Validate -> Preprocess -> Store
"""

import unittest
import pandas as pd
import os
import tempfile
import shutil
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import DataLoader
from data_validator import DataValidator
from data_preprocessor import DataPreprocessor
from data_storage import DataStorage


class TestPhase1Integration(unittest.TestCase):
    """Integration tests for Phase 1 Step 1"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock dataset
        self.mock_df = pd.DataFrame({
            'url': ['url1', 'url2', 'url3'],
            'address': ['Addr1', 'Addr2', 'Addr3'],
            'name': ['Restaurant A', 'Restaurant B', 'Restaurant C'],
            'online_order': ['Yes', 'No', 'Yes'],
            'book_table': ['Yes', 'Yes', 'No'],
            'rate': ['4.1/5', '4.5/5', '3.8/5'],
            'votes': [100, 200, 150],
            'phone': ['123', '456', '789'],
            'location': ['City A', 'City B', 'City A'],
            'rest_type': ['Casual', 'Fine', 'Casual'],
            'dish_liked': ['Dish1, Dish2', 'Dish3', 'Dish4, Dish5'],
            'cuisines': ['North Indian, Chinese', 'Italian', 'Mexican'],
            'approx_cost(for two people)': ['800', '1000', '600'],
            'reviews_list': [
                "[('Rated 4.0', 'Good')]",
                "[('Rated 5.0', 'Excellent')]",
                "[('Rated 3.0', 'Average')]"
            ],
            'menu_item': ['Item1', 'Item2', 'Item3'],
            'listed_in(type)': ['Buffet', 'Dine-out', 'Buffet'],
            'listed_in(city)': ['City A', 'City B', 'City A']
        })
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir)
    
    @patch('data_loader.load_dataset')
    def test_complete_pipeline(self, mock_load_dataset):
        """Test complete pipeline: Load -> Validate -> Preprocess -> Store"""
        # Mock dataset loading
        mock_dataset = MagicMock()
        mock_train = MagicMock()
        mock_train.to_pandas.return_value = self.mock_df
        mock_dataset.__getitem__.return_value = mock_train
        mock_dataset.keys.return_value = ['train']
        mock_load_dataset.return_value = mock_dataset
        
        # Step 1: Load
        loader = DataLoader()
        raw_df = loader.load_dataset()
        self.assertIsNotNone(raw_df)
        self.assertEqual(len(raw_df), 3)
        
        # Step 2: Validate
        validator = DataValidator(raw_df)
        validation_report = validator.validate_all()
        self.assertIsNotNone(validation_report)
        self.assertIn('summary', validation_report)
        
        # Step 3: Preprocess
        preprocessor = DataPreprocessor(raw_df)
        processed_df = preprocessor.preprocess_all()
        self.assertIsNotNone(processed_df)
        self.assertIn('restaurant_id', processed_df.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['rate']))
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['price']))
        
        # Step 4: Store
        storage = DataStorage(output_dir=self.test_dir)
        csv_path = storage.save_to_csv(processed_df, 'test_output.csv')
        self.assertTrue(os.path.exists(csv_path))
        
        # Verify stored data can be loaded
        loaded_df = storage.load_from_csv('test_output.csv')
        self.assertEqual(len(loaded_df), 3)
        self.assertIn('restaurant_id', loaded_df.columns)
    
    @patch('data_loader.load_dataset')
    def test_pipeline_with_missing_values(self, mock_load_dataset):
        """Test pipeline handles missing values correctly"""
        # Create DataFrame with missing values
        df_with_missing = self.mock_df.copy()
        df_with_missing.loc[0, 'rate'] = None
        df_with_missing.loc[1, 'price'] = None
        
        mock_dataset = MagicMock()
        mock_train = MagicMock()
        mock_train.to_pandas.return_value = df_with_missing
        mock_dataset.__getitem__.return_value = mock_train
        mock_dataset.keys.return_value = ['train']
        mock_load_dataset.return_value = mock_dataset
        
        # Run pipeline
        loader = DataLoader()
        raw_df = loader.load_dataset()
        
        validator = DataValidator(raw_df)
        validation_report = validator.validate_all()
        # Should still validate (missing values are acceptable)
        
        preprocessor = DataPreprocessor(raw_df)
        processed_df = preprocessor.preprocess_all()
        # Should handle missing values gracefully
        self.assertIsNotNone(processed_df)
    
    @patch('data_loader.load_dataset')
    def test_pipeline_data_structure(self, mock_load_dataset):
        """Test that final data structure matches expected format"""
        mock_dataset = MagicMock()
        mock_train = MagicMock()
        mock_train.to_pandas.return_value = self.mock_df
        mock_dataset.__getitem__.return_value = mock_train
        mock_dataset.keys.return_value = ['train']
        mock_load_dataset.return_value = mock_dataset
        
        loader = DataLoader()
        raw_df = loader.load_dataset()
        
        preprocessor = DataPreprocessor(raw_df)
        processed_df = preprocessor.preprocess_all()
        final_df = preprocessor.get_final_structure()
        
        # Check expected columns exist
        expected_columns = [
            'restaurant_id', 'name', 'location', 'city', 'rate', 'votes',
            'price', 'cuisines', 'rest_type', 'online_order', 'book_table'
        ]
        
        for col in expected_columns:
            self.assertIn(col, final_df.columns, f"Column {col} missing")
        
        # Check data types
        self.assertTrue(pd.api.types.is_integer_dtype(final_df['restaurant_id']))
        self.assertTrue(pd.api.types.is_numeric_dtype(final_df['rate']))
        self.assertTrue(pd.api.types.is_numeric_dtype(final_df['price']))
        self.assertTrue(pd.api.types.is_bool_dtype(final_df['online_order']))


if __name__ == '__main__':
    unittest.main()
