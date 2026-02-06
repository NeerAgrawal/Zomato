"""
Data Integration Module - STEP 3
Purpose: Filter restaurants and engineer features for recommendation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFilter:
    """Filter restaurants based on user criteria"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize data filter
        
        Args:
            df: Processed restaurant DataFrame
        """
        self.df = df
    
    def filter_by_city(self, city: str) -> pd.DataFrame:
        """
        Filter restaurants by city
        
        Args:
            city: City name to filter by
            
        Returns:
            pandas.DataFrame: Filtered DataFrame
        """
        logger.info(f"Filtering by city: {city}")
        
        city_lower = city.lower()
        
        # Match on both 'city' and 'location' columns
        mask = pd.Series([False] * len(self.df), index=self.df.index)
        
        if 'city' in self.df.columns:
            mask |= self.df['city'].str.lower() == city_lower
        
        if 'location' in self.df.columns:
            mask |= self.df['location'].str.lower() == city_lower
        
        filtered = self.df[mask].copy()
        logger.info(f"Found {len(filtered)} restaurants in {city}")
        
        return filtered
    
    def filter_by_price(self, df: pd.DataFrame, max_price: float) -> pd.DataFrame:
        """
        Filter restaurants by price
        
        Args:
            df: DataFrame to filter
            max_price: Maximum price for two people
            
        Returns:
            pandas.DataFrame: Filtered DataFrame
        """
        logger.info(f"Filtering by price: <= INR {max_price}")
        
        if 'price' not in df.columns:
            logger.warning("Price column not found, skipping price filter")
            return df
        
        # Filter where price <= max_price (handle NaN)
        filtered = df[df['price'].notna() & (df['price'] <= max_price)].copy()
        
        logger.info(f"Found {len(filtered)} restaurants within budget")
        
        return filtered
    
    def apply_filters(self, city: str, max_price: float) -> pd.DataFrame:
        """
        Apply all filters
        
        Args:
            city: City name
            max_price: Maximum price
            
        Returns:
            pandas.DataFrame: Filtered DataFrame
        """
        filtered = self.filter_by_city(city)
        filtered = self.filter_by_price(filtered, max_price)
        
        return filtered


class FeatureEngineer:
    """Engineer features for recommendation"""
    
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create scoring features for recommendations
        
        Args:
            df: Filtered DataFrame
            
        Returns:
            pandas.DataFrame: DataFrame with additional features
        """
        logger.info("Engineering features for recommendation...")
        
        df = df.copy()
        
        # Rating score (normalized 0-1)
        if 'rate' in df.columns:
            max_rate = df['rate'].max() if df['rate'].max() > 0 else 5.0
            df['rating_score'] = df['rate'].fillna(0) / max_rate
        else:
            df['rating_score'] = 0.5
        
        # Popularity score (log-normalized votes)
        if 'votes' in df.columns:
            df['popularity_score'] = np.log1p(df['votes'].fillna(0))
            max_pop = df['popularity_score'].max()
            if max_pop > 0:
                df['popularity_score'] = df['popularity_score'] / max_pop
        else:
            df['popularity_score'] = 0.5
        
        # Price score (inverse - lower price = higher score)
        if 'price' in df.columns:
            max_price = df['price'].max()
            if max_price > 0:
                df['price_score'] = 1 - (df['price'].fillna(max_price) / max_price)
            else:
                df['price_score'] = 0.5
        else:
            df['price_score'] = 0.5
        
        # Completeness score (based on available data)
        completeness_cols = ['rate', 'votes', 'price', 'cuisines', 'rest_type', 'dish_liked']
        available_cols = [col for col in completeness_cols if col in df.columns]
        
        if available_cols:
            df['completeness_score'] = df[available_cols].notna().sum(axis=1) / len(available_cols)
        else:
            df['completeness_score'] = 0.5
        
        # Combined score (weighted average)
        df['recommendation_score'] = (
            0.4 * df['rating_score'] +
            0.3 * df['popularity_score'] +
            0.2 * df['price_score'] +
            0.1 * df['completeness_score']
        )
        
        logger.info(f"Feature engineering completed for {len(df)} restaurants")
        
        return df


class DataIntegrator:
    """Main integration pipeline"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize data integrator
        
        Args:
            df: Processed restaurant DataFrame
        """
        self.df = df
        self.filter = DataFilter(df)
        self.engineer = FeatureEngineer()
    
    def integrate(self, user_input: Dict) -> pd.DataFrame:
        """
        Integrate user input with data
        
        Args:
            user_input: Dictionary with 'city' and 'price'
            
        Returns:
            pandas.DataFrame: Filtered and enhanced DataFrame
        """
        logger.info("Starting data integration...")
        
        # Apply filters
        filtered = self.filter.apply_filters(
            user_input['city'],
            user_input['price']
        )
        
        if len(filtered) == 0:
            logger.warning("No restaurants found matching criteria")
            return filtered
        
        # Engineer features
        enhanced = self.engineer.engineer_features(filtered)
        
        # Deduplicate based on name and location (keep entry with highest completeness/rating if possible, but distinct first)
        enhanced = enhanced.drop_duplicates(subset=['name', 'location'], keep='first')
        
        # Sort by recommendation score
        enhanced = enhanced.sort_values('recommendation_score', ascending=False)
        
        logger.info(f"Integration completed: {len(enhanced)} restaurants ready for recommendation")
        
        return enhanced
    
    def get_statistics(self, filtered_df: pd.DataFrame) -> Dict:
        """
        Get statistics about filtered data
        
        Args:
            filtered_df: Filtered DataFrame
            
        Returns:
            dict: Statistics
        """
        if len(filtered_df) == 0:
            return {
                'total_restaurants': 0,
                'avg_rating': 0,
                'avg_price': 0,
                'price_range': {'min': 0, 'max': 0}
            }
        
        stats = {
            'total_restaurants': len(filtered_df),
            'avg_rating': filtered_df['rate'].mean() if 'rate' in filtered_df.columns else 0,
            'avg_price': filtered_df['price'].mean() if 'price' in filtered_df.columns else 0,
            'price_range': {
                'min': filtered_df['price'].min() if 'price' in filtered_df.columns else 0,
                'max': filtered_df['price'].max() if 'price' in filtered_df.columns else 0
            }
        }
        
        return stats
