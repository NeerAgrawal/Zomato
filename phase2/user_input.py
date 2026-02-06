"""
User Input Module - STEP 2
Purpose: Capture and validate user preferences (city and price)
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserInputCollector:
    """Collect user input via CLI"""
    
    @staticmethod
    def get_user_input() -> Dict[str, any]:
        """
        Collect city and price from user via CLI
        
        Returns:
            dict: User input with 'city' and 'price' keys
        """
        print("\n" + "="*60)
        print("ZOMATO RESTAURANT RECOMMENDATION SYSTEM")
        print("="*60)
        
        city = input("\nEnter city name: ").strip()
        
        while True:
            try:
                price_input = input("Enter maximum price for two people (INR): ").strip()
                price = float(price_input)
                if price <= 0:
                    print("X Price must be positive. Please try again.")
                    continue
                break
            except ValueError:
                print("X Invalid price. Please enter a number.")
        
        return {
            'city': city,
            'price': price
        }


class UserInputValidator:
    """Validate user inputs against dataset"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize validator with dataset
        
        Args:
            df: Processed restaurant DataFrame
        """
        self.df = df
        self.available_cities = self._get_available_cities()
    
    def _get_available_cities(self) -> List[str]:
        """Get list of unique cities from dataset"""
        cities = set()
        
        if 'city' in self.df.columns:
            cities.update(self.df['city'].dropna().unique())
        if 'location' in self.df.columns:
            cities.update(self.df['location'].dropna().unique())
        
        return sorted([city for city in cities if city])
    
    def validate_city(self, city: str) -> Tuple[bool, Optional[str], Optional[List[str]]]:
        """
        Validate city exists in dataset
        
        Args:
            city: User-provided city name
            
        Returns:
            tuple: (is_valid, normalized_city, suggestions)
        """
        if not city:
            return False, None, self.available_cities[:10]
        
        # Exact match (case-insensitive)
        city_lower = city.lower()
        for available_city in self.available_cities:
            if available_city.lower() == city_lower:
                return True, available_city, None
        
        # Partial match
        suggestions = [
            c for c in self.available_cities 
            if city_lower in c.lower()
        ][:5]
        
        return False, None, suggestions if suggestions else self.available_cities[:10]
    
    def validate_price(self, price: float) -> Tuple[bool, Optional[str]]:
        """
        Validate price is reasonable
        
        Args:
            price: User-provided maximum price
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if price <= 0:
            return False, "Price must be positive"
        
        if price < 100:
            return False, "Price seems too low. Minimum recommended: ₹100"
        
        if price > 10000:
            return False, "Price seems too high. Maximum recommended: ₹10,000"
        
        return True, None
    
    def validate_all(self, user_input: Dict) -> Tuple[bool, Dict]:
        """
        Validate all user inputs
        
        Args:
            user_input: Dictionary with 'city' and 'price'
            
        Returns:
            tuple: (is_valid, validation_result)
        """
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'normalized_city': None
        }
        
        # Validate city
        city_valid, normalized_city, suggestions = self.validate_city(user_input['city'])
        if not city_valid:
            result['is_valid'] = False
            result['errors'].append(f"City '{user_input['city']}' not found in dataset")
            result['suggestions'] = suggestions
        else:
            result['normalized_city'] = normalized_city
        
        # Validate price
        price_valid, price_error = self.validate_price(user_input['price'])
        if not price_valid:
            result['is_valid'] = False
            result['errors'].append(price_error)
        
        return result['is_valid'], result


class UserInputProcessor:
    """Process and normalize user inputs"""
    
    @staticmethod
    def normalize_input(user_input: Dict, normalized_city: Optional[str] = None) -> Dict:
        """
        Normalize and format user inputs
        
        Args:
            user_input: Raw user input
            normalized_city: Validated city name
            
        Returns:
            dict: Normalized input
        """
        return {
            'city': normalized_city or user_input['city'].strip().title(),
            'price': int(user_input['price']),
            'original_city': user_input['city']
        }


def get_validated_user_input(df: pd.DataFrame) -> Optional[Dict]:
    """
    Main function to get and validate user input
    
    Args:
        df: Processed restaurant DataFrame
        
    Returns:
        dict: Validated and normalized user input, or None if validation fails
    """
    collector = UserInputCollector()
    validator = UserInputValidator(df)
    processor = UserInputProcessor()
    
    # Collect input
    user_input = collector.get_user_input()
    
    # Validate input
    is_valid, validation_result = validator.validate_all(user_input)
    
    if not is_valid:
        print("\n[X] Validation Errors:")
        for error in validation_result['errors']:
            print(f"   - {error}")
        
        if 'suggestions' in validation_result:
            print("\n[*] Available cities (suggestions):")
            for city in validation_result['suggestions']:
                print(f"   - {city}")
        
        return None
    
    # Process input
    normalized_input = processor.normalize_input(
        user_input, 
        validation_result.get('normalized_city')
    )
    
    logger.info(f"User input validated: City={normalized_input['city']}, Price=₹{normalized_input['price']}")
    
    return normalized_input
