"""
Data Preprocessing Module - STEP 1.3
Purpose: Clean and transform raw data
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional
import logging
import ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess Zomato dataset"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataPreprocessor
        
        Args:
            df: Raw DataFrame to preprocess
        """
        self.df = df.copy()
        self.preprocessing_stats = {}
    
    def preprocess_all(self) -> pd.DataFrame:
        """
        Run all preprocessing steps
        
        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")
        
        # Add restaurant_id
        self.df['restaurant_id'] = range(1, len(self.df) + 1)
        
        # Preprocessing steps
        self.normalize_rate()
        self.parse_price()
        self.standardize_location()
        self.clean_text_fields()
        self.handle_missing_values()
        self.extract_features()
        
        logger.info("Preprocessing completed")
        return self.df
    
    def normalize_rate(self) -> None:
        """
        Convert rate from "4.1/5" format to float
        """
        logger.info("Normalizing rate column...")
        
        def extract_rate(rate_str):
            if pd.isna(rate_str) or rate_str == '':
                return np.nan
            
            # Extract numeric value from strings like "4.1/5" or "4.1"
            match = re.search(r'(\d+\.?\d*)', str(rate_str))
            if match:
                return float(match.group(1))
            return np.nan
        
        self.df['rate'] = self.df['rate'].apply(extract_rate)
        
        # Count normalized rates
        normalized_count = self.df['rate'].notna().sum()
        self.preprocessing_stats['rate_normalized'] = normalized_count
    
    def parse_price(self) -> None:
        """
        Convert price strings to numeric values
        """
        logger.info("Parsing price column...")
        
        def extract_price(price_str):
            if pd.isna(price_str) or price_str == '':
                return np.nan
            
            # Remove commas and extract numeric value
            price_str = str(price_str).replace(',', '')
            match = re.search(r'(\d+)', price_str)
            if match:
                return int(match.group(1))
            return np.nan
        
        self.df['price'] = self.df['approx_cost(for two people)'].apply(extract_price)
        
        # Count parsed prices
        parsed_count = self.df['price'].notna().sum()
        self.preprocessing_stats['price_parsed'] = parsed_count
    
    def standardize_location(self) -> None:
        """
        Normalize city/location names
        """
        logger.info("Standardizing location names...")
        
        def clean_location(loc_str):
            if pd.isna(loc_str):
                return None
            # Title case, strip whitespace
            return str(loc_str).strip().title()
        
        self.df['location'] = self.df['location'].apply(clean_location)
        self.df['city'] = self.df['listed_in(city)'].apply(clean_location)
        
        # Fill missing city with location
        self.df['city'] = self.df['city'].fillna(self.df['location'])
    
    def clean_text_fields(self) -> None:
        """
        Clean text fields: cuisines, dish_liked, reviews_list
        """
        logger.info("Cleaning text fields...")
        
        def parse_list_field(field_str):
            """Parse comma-separated strings into lists"""
            if pd.isna(field_str) or field_str == '':
                return []
            
            # Split by comma and clean
            items = [item.strip() for item in str(field_str).split(',')]
            return [item for item in items if item]
        
        # Parse cuisines and dish_liked
        self.df['cuisines'] = self.df['cuisines'].apply(parse_list_field)
        self.df['dish_liked'] = self.df['dish_liked'].apply(parse_list_field)
        
        # Parse reviews_list (it's stored as string representation of list)
        def parse_reviews(reviews_str):
            if pd.isna(reviews_str) or reviews_str == '':
                return []
            
            try:
                # Try to evaluate as Python literal
                reviews = ast.literal_eval(str(reviews_str))
                if isinstance(reviews, list):
                    return reviews
                return []
            except:
                return []
        
        self.df['reviews'] = self.df['reviews_list'].apply(parse_reviews)
        
        # Parse menu_item
        self.df['menu_items'] = self.df['menu_item'].apply(parse_list_field)
    
    def handle_missing_values(self) -> None:
        """
        Handle missing values
        """
        logger.info("Handling missing values...")
        
        # Fill missing boolean-like fields
        self.df['online_order'] = self.df['online_order'].fillna('No')
        self.df['book_table'] = self.df['book_table'].fillna('No')
        
        # Convert to boolean
        self.df['online_order'] = self.df['online_order'].str.lower() == 'yes'
        self.df['book_table'] = self.df['book_table'].str.lower() == 'yes'
        
        # Fill missing rest_type with 'Unknown'
        self.df['rest_type'] = self.df['rest_type'].fillna('Unknown')
        
        # Fill missing phone with empty string
        self.df['phone'] = self.df['phone'].fillna('')
        
        missing_before = self.df.isnull().sum().sum()
        self.preprocessing_stats['missing_values_before'] = int(missing_before)
        self.preprocessing_stats['missing_values_after'] = int(self.df.isnull().sum().sum())
    
    def extract_features(self) -> None:
        """
        Extract additional features from existing data
        """
        logger.info("Extracting features...")
        
        # Extract number of reviews
        self.df['num_reviews'] = self.df['reviews'].apply(len)
        
        # Extract average rating from reviews (if available)
        def extract_avg_review_rating(reviews):
            if not reviews:
                return np.nan
            
            ratings = []
            for review in reviews:
                if isinstance(review, tuple) and len(review) >= 1:
                    rating_str = str(review[0])
                    match = re.search(r'(\d+\.?\d*)', rating_str)
                    if match:
                        ratings.append(float(match.group(1)))
            
            return np.mean(ratings) if ratings else np.nan
        
        self.df['avg_review_rating'] = self.df['reviews'].apply(extract_avg_review_rating)
        
        # Count number of cuisines
        self.df['num_cuisines'] = self.df['cuisines'].apply(len)
        
        # Count number of popular dishes
        self.df['num_dishes'] = self.df['dish_liked'].apply(len)
    
    def get_preprocessed_dataframe(self) -> pd.DataFrame:
        """
        Get the preprocessed DataFrame
        
        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self.df
    
    def get_stats(self) -> Dict:
        """
        Get preprocessing statistics
        
        Returns:
            dict: Preprocessing statistics
        """
        return self.preprocessing_stats
    
    def get_final_structure(self) -> pd.DataFrame:
        """
        Get final structured data with selected columns
        
        Returns:
            pandas.DataFrame: Final structured DataFrame
        """
        final_columns = [
            'restaurant_id', 'name', 'location', 'city', 'rate', 'votes',
            'price', 'cuisines', 'rest_type', 'online_order', 'book_table',
            'dish_liked', 'address', 'phone', 'reviews', 'menu_items',
            'num_reviews', 'avg_review_rating', 'num_cuisines', 'num_dishes'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in final_columns if col in self.df.columns]
        return self.df[available_columns]
