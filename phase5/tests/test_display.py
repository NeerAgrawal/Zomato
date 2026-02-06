"""
Test cases for Display module
"""

import unittest
import pandas as pd
import sys
import os
from io import StringIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase5.display import CLIDisplay


class TestCLIDisplay(unittest.TestCase):
    """Test cases for CLIDisplay"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_recommendations = pd.DataFrame({
            'name': ['Restaurant A', 'Restaurant B'],
            'rate': [4.5, 4.0],
            'votes': [100, 200],
            'price': [800, 1000],
            'location': ['Banashankari', 'Basavanagudi'],
            'city': ['Banashankari', 'Basavanagudi'],
            'cuisines': [['North Indian', 'Chinese'], ['Italian']],
            'rest_type': ['Casual Dining', 'Fine Dining'],
            'online_order': [True, False],
            'book_table': [False, True],
            'dish_liked': [['Biryani', 'Paneer'], ['Pizza']],
            'phone': ['1234567890', '0987654321'],
            'reason': ['Great food and ambiance', 'Excellent service']
        })
        
        self.user_input = {'city': 'Banashankari', 'price': 1000}
    
    def test_display_recommendations(self):
        """Test displaying recommendations"""
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            CLIDisplay.display_recommendations(self.sample_recommendations, self.user_input)
            output = sys.stdout.getvalue()
            
            # Check that output contains key information
            self.assertIn('Restaurant A', output)
            self.assertIn('Restaurant B', output)
            self.assertIn('Banashankari', output)
            self.assertIn('4.5', output)
        finally:
            sys.stdout = old_stdout
    
    def test_display_empty_recommendations(self):
        """Test displaying empty recommendations"""
        empty_df = pd.DataFrame()
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            CLIDisplay.display_recommendations(empty_df, self.user_input)
            output = sys.stdout.getvalue()
            
            self.assertIn('NO RESTAURANTS FOUND', output)
        finally:
            sys.stdout = old_stdout
    
    def test_display_statistics(self):
        """Test displaying statistics"""
        stats = {
            'total_restaurants': 10,
            'avg_rating': 4.2,
            'avg_price': 850,
            'price_range': {'min': 600, 'max': 1200}
        }
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            CLIDisplay.display_statistics(stats)
            output = sys.stdout.getvalue()
            
            self.assertIn('STATISTICS', output)
            self.assertIn('10', output)
            self.assertIn('4.2', output)
        finally:
            sys.stdout = old_stdout
    
    def test_display_error(self):
        """Test displaying error message"""
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            CLIDisplay.display_error("Test error message")
            output = sys.stdout.getvalue()
            
            self.assertIn('ERROR', output)
            self.assertIn('Test error message', output)
        finally:
            sys.stdout = old_stdout


if __name__ == '__main__':
    unittest.main()
