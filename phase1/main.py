"""
Main orchestrator for Phase 1 - STEP 1: Input the Zomato Data
"""

import logging
import os
import json
from data_loader import DataLoader
from data_validator import DataValidator
from data_preprocessor import DataPreprocessor
from data_storage import DataStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_phase1_step1():
    """
    Execute Phase 1, Step 1: Load, validate, preprocess, and store Zomato data
    """
    logger.info("=" * 60)
    logger.info("Starting Phase 1 - STEP 1: Input the Zomato Data")
    logger.info("=" * 60)
    
    # Step 1.1: Load data
    logger.info("\n--- STEP 1.1: Loading Data ---")
    loader = DataLoader()
    raw_df = loader.load_dataset()
    dataset_info = loader.get_info()
    logger.info(f"Dataset Info: {json.dumps(dataset_info, indent=2, default=str)}")
    
    # Step 1.2: Validate data
    logger.info("\n--- STEP 1.2: Validating Data ---")
    validator = DataValidator(raw_df)
    validation_report = validator.validate_all()
    
    # Print validation summary
    summary = validation_report['summary']
    logger.info(f"Validation Summary:")
    logger.info(f"  - Total Rows: {summary['total_rows']}")
    logger.info(f"  - Total Columns: {summary['total_columns']}")
    logger.info(f"  - Missing Values: {summary['total_missing_percentage']}%")
    logger.info(f"  - Duplicate Rows: {summary['duplicate_rows']}")
    logger.info(f"  - Is Valid: {summary['is_valid']}")
    
    if not summary['is_valid']:
        logger.warning("Dataset validation found issues. Continuing with preprocessing...")
    
    # Save validation report
    os.makedirs('data', exist_ok=True)
    with open('data/validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    logger.info("Validation report saved to: data/validation_report.json")
    
    # Step 1.3: Preprocess data
    logger.info("\n--- STEP 1.3: Preprocessing Data ---")
    preprocessor = DataPreprocessor(raw_df)
    processed_df = preprocessor.preprocess_all()
    preprocessing_stats = preprocessor.get_stats()
    
    logger.info(f"Preprocessing Statistics:")
    for key, value in preprocessing_stats.items():
        logger.info(f"  - {key}: {value}")
    
    # Get final structured data
    final_df = preprocessor.get_final_structure()
    logger.info(f"\nFinal DataFrame Shape: {final_df.shape}")
    logger.info(f"Final DataFrame Columns: {list(final_df.columns)}")
    
    # Step 1.4: Store data
    logger.info("\n--- STEP 1.4: Storing Data ---")
    storage = DataStorage(output_dir='data')
    
    # Save in multiple formats
    csv_path = storage.save_to_csv(final_df)
    parquet_path = storage.save_to_parquet(final_df)
    pickle_path = storage.save_to_pickle(final_df)
    
    logger.info(f"\nData saved to:")
    logger.info(f"  - CSV: {csv_path}")
    logger.info(f"  - Parquet: {parquet_path}")
    logger.info(f"  - Pickle: {pickle_path}")
    
    # Display sample of processed data
    logger.info("\n--- Sample Processed Data (First 3 rows) ---")
    sample_cols = ['restaurant_id', 'name', 'location', 'city', 'rate', 'votes', 'price']
    available_cols = [col for col in sample_cols if col in final_df.columns]
    logger.info(f"\n{final_df[available_cols].head(3).to_string()}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 - STEP 1 completed successfully!")
    logger.info("=" * 60)
    
    return {
        'raw_dataframe': raw_df,
        'processed_dataframe': final_df,
        'validation_report': validation_report,
        'preprocessing_stats': preprocessing_stats,
        'storage_paths': {
            'csv': csv_path,
            'parquet': parquet_path,
            'pickle': pickle_path
        }
    }


if __name__ == "__main__":
    try:
        results = run_phase1_step1()
        logger.info("\n✓ All steps completed successfully!")
    except Exception as e:
        logger.error(f"\n✗ Error occurred: {str(e)}", exc_info=True)
        raise
