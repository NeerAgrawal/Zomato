"""
Test cases for Data Integration module
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase3.data_integration import DataFilter, FeatureEngineer, DataIntegrator


class TestDataFilter(unittest.TestCase):
    """Test cases for DataFilter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_df = pd.DataFrame({
            'restaurant_id': [1, 2, 3, 4, 5],
            'name': ['Restaurant A', 'Restaurant B', 'Restaurant C', 'Restaurant D', 'Restaurant E'],
            'city': ['Banashankari', 'Basavanagudi', 'Banashankari', 'Koramangala', 'Banashankari'],
            'location': ['Banashankari', 'Basavanagudi', 'Banashankari', 'Koramangala', 'Banashankari'],
            'price': [800, 1000, 600, 1200, 700]
        })
        
        self.filter = DataFilter(self.sample_df)
    
    def test_filter_by_city(self):
        """Test filtering by city"""
        filtered = self.filter.filter_by_city('Banashankari')
        self.assertEqual(len(filtered), 3)
        self.assertTrue(all(filtered['city'] == 'Banashankari'))
    
    def test_filter_by_city_case_insensitive(self):
        """Test city filtering is case-insensitive"""
        filtered = self.filter.filter_by_city('banashankari')
        self.assertEqual(len(filtered), 3)
    
    def test_filter_by_price(self):
        """Test filtering by price"""
        filtered = self.filter.filter_by_price(self.sample_df, 800)
        self.assertEqual(len(filtered), 3)
        self.assertTrue(all(filtered['price'] <= 800))
    
    def test_apply_filters(self):
        """Test applying both filters"""
        filtered = self.filter.apply_filters('Banashankari', 800)
        self.assertEqual(len(filtered), 3)  # 3 restaurants in Banashankari with price <= 800


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_df = pd.DataFrame({
            'restaurant_id': [1, 2, 3],
            'name': ['Restaurant A', 'Restaurant B', 'Restaurant C'],
            'rate': [4.5, 4.0, 3.8],
            'votes': [100, 200, 50],
            'price': [800, 1000, 600]
        })
    
    def test_engineer_features(self):
        """Test feature engineering"""
        engineered = FeatureEngineer.engineer_features(self.sample_df)
        
        # Check that new features are created
        self.assertIn('rating_score', engineered.columns)
        self.assertIn('popularity_score', engineered.columns)
        self.assertIn('price_score', engineered.columns)
        self.assertIn('completeness_score', engineered.columns)
        self.assertIn('recommendation_score', engineered.columns)
        
        # Check that scores are normalized (0-1)
        self.assertTrue(all(engineered['rating_score'] >= 0))
        self.assertTrue(all(engineered['rating_score'] <= 1))
        self.assertTrue(all(engineered['recommendation_score'] >= 0))
        self.assertTrue(all(engineered['recommendation_score'] <= 1))


class TestDataIntegrator(unittest.TestCase):
    """Test cases for DataIntegrator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_df = pd.DataFrame({
            'restaurant_id': [1, 2, 3, 4],
            'name': ['Restaurant A', 'Restaurant B', 'Restaurant C', 'Restaurant D'],
            'city': ['Banashankari', 'Banashankari', 'Basavanagudi', 'Banashankari'],
            'location': ['Banashankari', 'Banashankari', 'Basavanagudi', 'Banashankari'],
            'rate': [4.5, 4.0, 3.8, 4.2],
            'votes': [100, 200, 50, 150],
            'price': [800, 1000, 600, 700]
        })
        
        self.integrator = DataIntegrator(self.sample_df)
    
    def test_integrate(self):
        """Test full integration pipeline"""
        user_input = {'city': 'Banashankari', 'price': 800}
        result = self.integrator.integrate(user_input)
        
        # Check that filtering worked
        self.assertGreater(len(result), 0)
        self.assertTrue(all(result['price'] <= 800))
        
        # Check that features were engineered
        self.assertIn('recommendation_score', result.columns)
    
    def test_get_statistics(self):
        """Test statistics generation"""
        stats = self.integrator.get_statistics(self.sample_df)
        
        self.assertEqual(stats['total_restaurants'], 4)
        self.assertGreater(stats['avg_rating'], 0)
        self.assertGreater(stats['avg_price'], 0)
        self.assertIn('min', stats['price_range'])
        self.assertIn('max', stats['price_range'])


if __name__ == '__main__':
    unittest.main()
