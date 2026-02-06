"""
Display Module - STEP 5
Purpose: Display recommendations in CLI format
"""

import pandas as pd
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIDisplay:
    """Display recommendations in CLI"""
    
    @staticmethod
    def display_recommendations(recommendations: pd.DataFrame, user_input: Dict) -> None:
        """
        Display recommendations to console
        
        Args:
            recommendations: DataFrame with recommended restaurants
            user_input: User preferences
        """
        if len(recommendations) == 0:
            print("\n" + "="*60)
            print("[X] NO RESTAURANTS FOUND")
            print("="*60)
            print(f"\nNo restaurants found in {user_input['city']} within INR {user_input['price']} budget.")
            print("\n[*] Try:")
            print("   - Increasing your budget")
            print("   - Trying a different city")
            return
        
        print("\n" + "="*60)
        print("[Dining] RESTAURANT RECOMMENDATIONS")
        print("="*60)
        print(f"\nCity: {user_input['city']}")
        print(f"Budget: INR {user_input['price']} for two")
        print(f"Found: {len(recommendations)} recommendations")
        print("\n" + "-"*60)
        
        for idx, (_, restaurant) in enumerate(recommendations.iterrows(), 1):
            CLIDisplay._display_restaurant_card(idx, restaurant)
    
    @staticmethod
    def _display_restaurant_card(rank: int, restaurant: pd.Series) -> None:
        """
        Display individual restaurant card
        
        Args:
            rank: Rank number
            restaurant: Restaurant data
        """
        print(f"\n{rank}. {restaurant.get('name', 'Unknown Restaurant')}")
        print("   " + "-"*55)
        
        # Rating and votes
        rate = restaurant.get('rate', 'N/A')
        votes = restaurant.get('votes', 0)
        if rate != 'N/A':
            stars = "*" * int(float(rate))
            print(f"   Rating: {rate}/5 {stars} ({votes} votes)")
        else:
            print(f"   Rating: N/A ({votes} votes)")
        
        # Price
        price = restaurant.get('price', 'N/A')
        print(f"   Price: INR {price} for two")
        
        # Location
        location = restaurant.get('location', 'N/A')
        city = restaurant.get('city', '')
        if city and city != location:
            print(f"   Location: {location}, {city}")
        else:
            print(f"   Location: {location}")
        
        # Cuisines
        cuisines = restaurant.get('cuisines', [])
        if isinstance(cuisines, list) and cuisines:
            cuisines_str = ', '.join(cuisines[:5])  # Limit to 5
            print(f"   Cuisines: {cuisines_str}")
        elif cuisines:
            print(f"   Cuisines: {cuisines}")
        
        # Restaurant type
        rest_type = restaurant.get('rest_type', 'N/A')
        print(f"   Type: {rest_type}")
        
        # Features
        features = []
        if restaurant.get('online_order'):
            features.append("[Delivery] Online Order")
        if restaurant.get('book_table'):
            features.append("[Booking] Table Booking")
        
        if features:
            print(f"   Features: {' | '.join(features)}")
        
        # Popular dishes
        dishes = restaurant.get('dish_liked', [])
        if isinstance(dishes, list) and dishes:
            dishes_str = ', '.join(dishes[:3])  # Limit to 3
            print(f"   Popular Dishes: {dishes_str}")
        
        # Contact
        phone = restaurant.get('phone', '')
        if phone:
            print(f"   Phone: {phone}")
        
        # Recommendation reason (from LLM)
        reason = restaurant.get('reason', '')
        if reason:
            print(f"   [*] Why recommended: {reason}")
        
        print()
    
    @staticmethod
    def display_statistics(stats: Dict) -> None:
        """
        Display statistics about recommendations
        
        Args:
            stats: Statistics dictionary
        """
        print("\n" + "="*60)
        print("[Stats] STATISTICS")
        print("="*60)
        
        print(f"\nTotal Restaurants: {stats.get('total_restaurants', 0)}")
        print(f"Average Rating: {stats.get('avg_rating', 0):.2f}/5")
        print(f"Average Price: INR {stats.get('avg_price', 0):.0f}")
        
        price_range = stats.get('price_range', {})
        print(f"Price Range: INR {price_range.get('min', 0):.0f} - INR {price_range.get('max', 0):.0f}")
        
        print("\n" + "="*60)
    
    @staticmethod
    def display_error(message: str) -> None:
        """
        Display error message
        
        Args:
            message: Error message
        """
        print("\n" + "="*60)
        print("[X] ERROR")
        print("="*60)
        print(f"\n{message}")
        print("\n" + "="*60)
