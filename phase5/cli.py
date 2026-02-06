"""
Main orchestrator for Phase 2 - User Input, Integration, and Recommendation
"""

import logging
import sys
import os
import pandas as pd
from pathlib import Path

# Add parent directory to path to import from phase1
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase2.user_input import get_validated_user_input
from phase3.data_integration import DataIntegrator
from phase4.engine import GroqEngine
from phase3.data_integration import DataIntegrator
from phase4.engine import GroqEngine
from phase5.display import CLIDisplay
from dotenv import load_dotenv

# Load environment variables from phase4/.env
env_path = Path(__file__).parent.parent / 'phase4' / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Fallback to phase2/.env if phase4 doesn't exist
    load_dotenv(Path(__file__).parent.parent / 'phase2' / '.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_processed_data() -> pd.DataFrame:
    """
    Load processed data from Phase 1
    
    Returns:
        pandas.DataFrame: Processed restaurant data
    """
    # Try to load from pickle (preserves data types)
    data_path = Path(__file__).parent.parent / 'phase1' / 'data' / 'processed_zomato_data.pkl'
    
    if not data_path.exists():
        # Try CSV as fallback
        data_path = Path(__file__).parent.parent / 'phase1' / 'data' / 'processed_zomato_data.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(
            "Processed data not found. Please run Phase 1 first to generate processed data."
        )
    
    logger.info(f"Loading processed data from: {data_path}")
    
    if data_path.suffix == '.pkl':
        df = pd.read_pickle(data_path)
    else:
        df = pd.read_csv(data_path)
    
    logger.info(f"Loaded {len(df)} restaurants")
    
    return df


def run_phase2():
    """
    Execute Phase 2: User Input, Integration, and Recommendation
    """
    logger.info("="*60)
    logger.info("Starting Zomato Restaurant Recommendation System")
    logger.info("="*60)
    
    try:
        # Step 1: Load processed data from Phase 1
        logger.info("\n--- STEP 1: Loading Data ---")
        df = load_processed_data()
        
        # Step 2: Get user input
        logger.info("\n--- STEP 2: User Input ---")
        user_input = get_validated_user_input(df)
        
        if user_input is None:
            logger.error("User input validation failed")
            return
        
        # Step 3: Integrate data (filter and engineer features)
        logger.info("\n--- STEP 3: Data Integration ---")
        integrator = DataIntegrator(df)
        filtered_df = integrator.integrate(user_input)
        
        if len(filtered_df) == 0:
            CLIDisplay.display_error(
                f"No restaurants found in {user_input['city']} within ₹{user_input['price']} budget.\n"
                "Please try a different city or increase your budget."
            )
            return
        
        # Get statistics
        stats = integrator.get_statistics(filtered_df)
        
        # Step 4: Generate recommendations
        logger.info("\n--- STEP 4: Recommendation ---")
        recommender = GroqEngine()
        recommendations = recommender.generate_recommendations(
            filtered_df, 
            user_input, 
            top_k=10
        )
        
        # Step 5: Display recommendations
        logger.info("\n--- STEP 5: Display ---")
        CLIDisplay.display_recommendations(recommendations, user_input)
        CLIDisplay.display_statistics(stats)
        
        logger.info("\n" + "="*60)
        logger.info("✓ Recommendation process completed successfully!")
        logger.info("="*60)
        
        return {
            'user_input': user_input,
            'filtered_count': len(filtered_df),
            'recommendations': recommendations,
            'statistics': stats
        }
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        CLIDisplay.display_error(str(e))
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        CLIDisplay.display_error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        results = run_phase2()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
