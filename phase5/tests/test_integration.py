"""
Integration tests for Phase 2 pipeline
"""

import unittest
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.user_input import UserInputValidator, UserInputProcessor
from phase3.data_integration import DataIntegrator
from phase4.engine import GroqEngine


class TestPhase2Integration(unittest.TestCase):
    """Integration tests for Phase 2"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_df = pd.DataFrame({
            'restaurant_id': list(range(1, 21)),
            'name': [f'Restaurant {chr(65+i)}' for i in range(20)],
            'city': ['Banashankari'] * 10 + ['Basavanagudi'] * 10,
            'location': ['Banashankari'] * 10 + ['Basavanagudi'] * 10,
            'rate': [4.0 + (i % 10) * 0.1 for i in range(20)],
            'votes': [100 + i * 10 for i in range(20)],
            'price': [600 + i * 50 for i in range(20)],
            'cuisines': [['North Indian', 'Chinese']] * 20,
            'rest_type': ['Casual Dining'] * 20,
            'online_order': [True] * 20,
            'book_table': [False] * 20,
            'dish_liked': [['Biryani', 'Paneer']] * 20
        })
    
    def test_complete_pipeline(self):
        """Test complete Phase 2 pipeline"""
        # Step 1: Validate user input
        user_input = {'city': 'Banashankari', 'price': 800}
        validator = UserInputValidator(self.sample_df)
        is_valid, result = validator.validate_all(user_input)
        
        self.assertTrue(is_valid)
        
        # Step 2: Normalize input
        normalized_input = UserInputProcessor.normalize_input(
            user_input, 
            result['normalized_city']
        )
        
        # Step 3: Integrate data
        integrator = DataIntegrator(self.sample_df)
        filtered_df = integrator.integrate(normalized_input)
        
        self.assertGreater(len(filtered_df), 0)
        self.assertIn('recommendation_score', filtered_df.columns)
        
        # Step 4: Generate recommendations
        recommender = GroqEngine(api_key=None)  # Use fallback
        recommendations = recommender.generate_recommendations(
            filtered_df, 
            normalized_input, 
            top_k=5
        )
        
        self.assertGreater(len(recommendations), 0)
        self.assertLessEqual(len(recommendations), 5)
        self.assertIn('reason', recommendations.columns)
    
    def test_pipeline_with_no_results(self):
        """Test pipeline when no restaurants match criteria"""
        user_input = {'city': 'Banashankari', 'price': 500}  # Very low price
        validator = UserInputValidator(self.sample_df)
        is_valid, result = validator.validate_all(user_input)
        
        self.assertTrue(is_valid)
        
        normalized_input = UserInputProcessor.normalize_input(
            user_input, 
            result['normalized_city']
        )
        
        integrator = DataIntegrator(self.sample_df)
        filtered_df = integrator.integrate(normalized_input)
        
        # Should have no results or very few
        self.assertLessEqual(len(filtered_df), 1)
    
    def test_pipeline_with_different_city(self):
        """Test pipeline with different city"""
        user_input = {'city': 'Basavanagudi', 'price': 1000}
        validator = UserInputValidator(self.sample_df)
        is_valid, result = validator.validate_all(user_input)
        
        self.assertTrue(is_valid)
        
        normalized_input = UserInputProcessor.normalize_input(
            user_input, 
            result['normalized_city']
        )
        
        integrator = DataIntegrator(self.sample_df)
        filtered_df = integrator.integrate(normalized_input)
        
        # All results should be from Basavanagudi
        self.assertTrue(all(filtered_df['city'] == 'Basavanagudi'))


if __name__ == '__main__':
    unittest.main()
