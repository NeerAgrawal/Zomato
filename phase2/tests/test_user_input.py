"""
Test cases for User Input module
"""

import unittest
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.user_input import UserInputValidator, UserInputProcessor


class TestUserInputValidator(unittest.TestCase):
    """Test cases for UserInputValidator"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample DataFrame
        self.sample_df = pd.DataFrame({
            'restaurant_id': [1, 2, 3],
            'name': ['Restaurant A', 'Restaurant B', 'Restaurant C'],
            'city': ['Banashankari', 'Basavanagudi', 'Koramangala'],
            'location': ['Banashankari', 'Basavanagudi', 'Koramangala'],
            'price': [800, 1000, 600]
        })
        
        self.validator = UserInputValidator(self.sample_df)
    
    def test_validate_city_exact_match(self):
        """Test city validation with exact match"""
        is_valid, normalized, suggestions = self.validator.validate_city('Banashankari')
        self.assertTrue(is_valid)
        self.assertEqual(normalized, 'Banashankari')
        self.assertIsNone(suggestions)
    
    def test_validate_city_case_insensitive(self):
        """Test city validation is case-insensitive"""
        is_valid, normalized, suggestions = self.validator.validate_city('banashankari')
        self.assertTrue(is_valid)
        self.assertEqual(normalized, 'Banashankari')
    
    def test_validate_city_invalid(self):
        """Test city validation with invalid city"""
        is_valid, normalized, suggestions = self.validator.validate_city('InvalidCity')
        self.assertFalse(is_valid)
        self.assertIsNone(normalized)
        self.assertIsNotNone(suggestions)
    
    def test_validate_city_partial_match(self):
        """Test city validation with partial match"""
        is_valid, normalized, suggestions = self.validator.validate_city('Bana')
        self.assertFalse(is_valid)
        self.assertIn('Banashankari', suggestions)
    
    def test_validate_price_valid(self):
        """Test price validation with valid price"""
        is_valid, error = self.validator.validate_price(800)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_validate_price_negative(self):
        """Test price validation with negative price"""
        is_valid, error = self.validator.validate_price(-100)
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
    
    def test_validate_price_too_low(self):
        """Test price validation with very low price"""
        is_valid, error = self.validator.validate_price(50)
        self.assertFalse(is_valid)
        self.assertIn('too low', error)
    
    def test_validate_price_too_high(self):
        """Test price validation with very high price"""
        is_valid, error = self.validator.validate_price(15000)
        self.assertFalse(is_valid)
        self.assertIn('too high', error)
    
    def test_validate_all_valid(self):
        """Test validation of all inputs with valid data"""
        user_input = {'city': 'Banashankari', 'price': 800}
        is_valid, result = self.validator.validate_all(user_input)
        self.assertTrue(is_valid)
        self.assertEqual(result['normalized_city'], 'Banashankari')
        self.assertEqual(len(result['errors']), 0)
    
    def test_validate_all_invalid_city(self):
        """Test validation with invalid city"""
        user_input = {'city': 'InvalidCity', 'price': 800}
        is_valid, result = self.validator.validate_all(user_input)
        self.assertFalse(is_valid)
        self.assertGreater(len(result['errors']), 0)


class TestUserInputProcessor(unittest.TestCase):
    """Test cases for UserInputProcessor"""
    
    def test_normalize_input(self):
        """Test input normalization"""
        user_input = {'city': '  banashankari  ', 'price': 800.5}
        normalized = UserInputProcessor.normalize_input(user_input, 'Banashankari')
        
        self.assertEqual(normalized['city'], 'Banashankari')
        self.assertEqual(normalized['price'], 800)
        self.assertIn('original_city', normalized)


if __name__ == '__main__':
    unittest.main()
