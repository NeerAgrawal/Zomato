"""
Test cases for DataPreprocessor module
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_df = pd.DataFrame({
            'name': ['Restaurant A', 'Restaurant B'],
            'rate': ['4.1/5', '4.5/5'],
            'votes': [100, 200],
            'approx_cost(for two people)': ['800', '1000'],
            'location': ['City A', 'City B'],
            'listed_in(city)': ['City A', 'City B'],
            'cuisines': ['North Indian, Chinese', 'Italian, Mexican'],
            'dish_liked': ['Dish1, Dish2', 'Dish3'],
            'reviews_list': ["[('Rated 4.0', 'Good')]", "[('Rated 5.0', 'Excellent')]"],
            'menu_item': ['Item1, Item2', 'Item3'],
            'online_order': ['Yes', 'No'],
            'book_table': ['Yes', 'Yes'],
            'rest_type': ['Casual Dining', 'Fine Dining'],
            'phone': ['123456', '789012'],
            'address': ['Addr1', 'Addr2'],
            'url': ['url1', 'url2']
        })
        self.preprocessor = DataPreprocessor(self.sample_df)
    
    def test_normalize_rate(self):
        """Test rate normalization"""
        self.preprocessor.normalize_rate()
        
        self.assertTrue(pd.api.types.is_numeric_dtype(self.preprocessor.df['rate']))
        self.assertAlmostEqual(self.preprocessor.df.loc[0, 'rate'], 4.1, places=1)
        self.assertAlmostEqual(self.preprocessor.df.loc[1, 'rate'], 4.5, places=1)
    
    def test_parse_price(self):
        """Test price parsing"""
        self.preprocessor.parse_price()
        
        self.assertTrue(pd.api.types.is_numeric_dtype(self.preprocessor.df['price']))
        self.assertEqual(self.preprocessor.df.loc[0, 'price'], 800)
        self.assertEqual(self.preprocessor.df.loc[1, 'price'], 1000)
    
    def test_standardize_location(self):
        """Test location standardization"""
        df_with_mixed_case = pd.DataFrame({
            'location': ['  city a  ', 'CITY B', 'city c'],
            'listed_in(city)': ['City A', 'city b', 'CITY C']
        })
        preprocessor = DataPreprocessor(df_with_mixed_case)
        preprocessor.standardize_location()
        
        self.assertEqual(preprocessor.df.loc[0, 'location'], 'City A')
        self.assertEqual(preprocessor.df.loc[1, 'location'], 'City B')
    
    def test_clean_text_fields(self):
        """Test text field cleaning"""
        self.preprocessor.clean_text_fields()
        
        # Check cuisines is a list
        self.assertIsInstance(self.preprocessor.df.loc[0, 'cuisines'], list)
        self.assertEqual(len(self.preprocessor.df.loc[0, 'cuisines']), 2)
        
        # Check dish_liked is a list
        self.assertIsInstance(self.preprocessor.df.loc[0, 'dish_liked'], list)
        self.assertEqual(len(self.preprocessor.df.loc[0, 'dish_liked']), 2)
        
        # Check reviews is a list
        self.assertIsInstance(self.preprocessor.df.loc[0, 'reviews'], list)
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        df_with_missing = self.sample_df.copy()
        df_with_missing.loc[0, 'online_order'] = np.nan
        df_with_missing.loc[1, 'book_table'] = np.nan
        df_with_missing.loc[0, 'rest_type'] = np.nan
        
        preprocessor = DataPreprocessor(df_with_missing)
        preprocessor.handle_missing_values()
        
        # Check boolean conversion
        self.assertIsInstance(preprocessor.df.loc[0, 'online_order'], (bool, np.bool_))
        self.assertIsInstance(preprocessor.df.loc[1, 'book_table'], (bool, np.bool_))
    
    def test_extract_features(self):
        """Test feature extraction"""
        self.preprocessor.clean_text_fields()
        self.preprocessor.extract_features()
        
        # Check num_reviews
        self.assertIn('num_reviews', self.preprocessor.df.columns)
        
        # Check num_cuisines
        self.assertIn('num_cuisines', self.preprocessor.df.columns)
        self.assertEqual(self.preprocessor.df.loc[0, 'num_cuisines'], 2)
        
        # Check num_dishes
        self.assertIn('num_dishes', self.preprocessor.df.columns)
        self.assertEqual(self.preprocessor.df.loc[0, 'num_dishes'], 2)
    
    def test_preprocess_all(self):
        """Test complete preprocessing"""
        processed_df = self.preprocessor.preprocess_all()
        
        # Check restaurant_id was added
        self.assertIn('restaurant_id', processed_df.columns)
        
        # Check rate is numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['rate']))
        
        # Check price is numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['price']))
        
        # Check boolean fields
        self.assertTrue(pd.api.types.is_bool_dtype(processed_df['online_order']))
        self.assertTrue(pd.api.types.is_bool_dtype(processed_df['book_table']))
    
    def test_get_stats(self):
        """Test preprocessing statistics"""
        self.preprocessor.preprocess_all()
        stats = self.preprocessor.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('rate_normalized', stats)
        self.assertIn('price_parsed', stats)
    
    def test_get_final_structure(self):
        """Test final structure retrieval"""
        self.preprocessor.preprocess_all()
        final_df = self.preprocessor.get_final_structure()
        
        self.assertIn('restaurant_id', final_df.columns)
        self.assertIn('name', final_df.columns)
        self.assertIn('rate', final_df.columns)
        self.assertIn('price', final_df.columns)


if __name__ == '__main__':
    unittest.main()
