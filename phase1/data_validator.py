"""
Data Validation Module - STEP 1.2
Purpose: Validate data integrity and completeness
"""

import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validate Zomato dataset integrity and completeness"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataValidator
        
        Args:
            df: DataFrame to validate
        """
        self.df = df
        self.validation_report = {}
    
    def validate_all(self) -> Dict:
        """
        Run all validation checks
        
        Returns:
            dict: Comprehensive validation report
        """
        logger.info("Starting data validation...")
        
        self.validation_report = {
            "missing_values": self.check_missing_values(),
            "data_types": self.check_data_types(),
            "duplicates": self.check_duplicates(),
            "schema": self.check_schema(),
            "summary": {}
        }
        
        # Calculate summary statistics
        total_rows = len(self.df)
        total_columns = len(self.df.columns)
        missing_percentage = (self.validation_report["missing_values"]["total_missing"] / 
                             (total_rows * total_columns)) * 100
        
        self.validation_report["summary"] = {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "total_missing_percentage": round(missing_percentage, 2),
            "duplicate_rows": self.validation_report["duplicates"]["count"],
            "is_valid": self.is_valid()
        }
        
        logger.info("Validation completed")
        return self.validation_report
    
    def check_missing_values(self) -> Dict:
        """
        Check for missing values in the dataset
        
        Returns:
            dict: Missing values report
        """
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        missing_report = {
            "by_column": {},
            "total_missing": int(missing_counts.sum()),
            "columns_with_missing": []
        }
        
        for col in self.df.columns:
            missing_count = int(missing_counts[col])
            missing_pct = round(missing_percentages[col], 2)
            
            missing_report["by_column"][col] = {
                "count": missing_count,
                "percentage": missing_pct
            }
            
            if missing_count > 0:
                missing_report["columns_with_missing"].append({
                    "column": col,
                    "count": missing_count,
                    "percentage": missing_pct
                })
        
        return missing_report
    
    def check_data_types(self) -> Dict:
        """
        Validate data types
        
        Returns:
            dict: Data type validation report
        """
        expected_types = {
            "url": "string",
            "address": "string",
            "name": "string",
            "online_order": "string",
            "book_table": "string",
            "rate": "string",
            "votes": "int64",
            "phone": "string",
            "location": "string",
            "rest_type": "string",
            "dish_liked": "string",
            "cuisines": "string",
            "approx_cost(for two people)": "string",
            "reviews_list": "string",
            "menu_item": "string",
            "listed_in(type)": "string",
            "listed_in(city)": "string"
        }
        
        type_report = {
            "actual_types": self.df.dtypes.to_dict(),
            "type_mismatches": [],
            "all_match": True
        }
        
        for col, expected_type in expected_types.items():
            if col in self.df.columns:
                actual_type = str(self.df[col].dtype)
                if expected_type not in actual_type.lower():
                    type_report["type_mismatches"].append({
                        "column": col,
                        "expected": expected_type,
                        "actual": actual_type
                    })
                    type_report["all_match"] = False
        
        return type_report
    
    def check_duplicates(self) -> Dict:
        """
        Check for duplicate rows
        
        Returns:
            dict: Duplicate rows report
        """
        duplicate_mask = self.df.duplicated()
        duplicate_count = int(duplicate_mask.sum())
        
        return {
            "count": duplicate_count,
            "percentage": round((duplicate_count / len(self.df)) * 100, 2) if len(self.df) > 0 else 0,
            "has_duplicates": duplicate_count > 0
        }
    
    def check_schema(self) -> Dict:
        """
        Validate dataset schema (required columns)
        
        Returns:
            dict: Schema validation report
        """
        required_columns = [
            "url", "address", "name", "online_order", "book_table",
            "rate", "votes", "phone", "location", "rest_type",
            "dish_liked", "cuisines", "approx_cost(for two people)",
            "reviews_list", "menu_item", "listed_in(type)", "listed_in(city)"
        ]
        
        actual_columns = set(self.df.columns)
        required_set = set(required_columns)
        
        missing_columns = required_set - actual_columns
        extra_columns = actual_columns - required_set
        
        return {
            "required_columns": required_columns,
            "actual_columns": list(actual_columns),
            "missing_columns": list(missing_columns),
            "extra_columns": list(extra_columns),
            "is_valid": len(missing_columns) == 0
        }
    
    def is_valid(self) -> bool:
        """
        Check if dataset passes all validation checks
        
        Returns:
            bool: True if dataset is valid
        """
        if not self.validation_report:
            return False
        
        schema_valid = self.validation_report["schema"]["is_valid"]
        no_critical_missing = (
            self.validation_report["missing_values"]["total_missing"] < len(self.df) * 0.5
        )
        
        return schema_valid and no_critical_missing
    
    def get_report(self) -> Dict:
        """
        Get validation report
        
        Returns:
            dict: Validation report
        """
        return self.validation_report
