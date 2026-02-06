import sys
from pathlib import Path
import os
import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path to import from sibling phases
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase3.data_integration import DataIntegrator
from phase4.engine import GroqEngine
from phase6.schemas import UserRequest, RecommendationResponse

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load Environment and Data
app = FastAPI(title="Zomato Recommendation API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Data Cache
PROCESSED_DATA_PATH_PKL = Path(__file__).parent.parent / 'phase1' / 'data' / 'processed_zomato_data.pkl'
PROCESSED_DATA_PATH_CSV = Path(__file__).parent.parent / 'phase1' / 'data' / 'processed_zomato_data.csv'
ENV_PATH = Path(__file__).parent.parent / 'phase4' / '.env'

class ServiceContainer:
    df: pd.DataFrame = None
    
    @classmethod
    def load_resources(cls):
        """Load data and env variables on startup"""
        # Load Env
        if ENV_PATH.exists():
            load_dotenv(ENV_PATH)
            logger.info("Loaded .env from phase4")
        else:
            logger.warning("phase4/.env not found, checking phase2")
            load_dotenv(Path(__file__).parent.parent / 'phase2' / '.env')
            
        # Load Data
        if PROCESSED_DATA_PATH_PKL.exists():
            cls.df = pd.read_pickle(PROCESSED_DATA_PATH_PKL)
            logger.info(f"Loaded {len(cls.df)} records from pickle")
        elif PROCESSED_DATA_PATH_CSV.exists():
            cls.df = pd.read_csv(PROCESSED_DATA_PATH_CSV)
            logger.info(f"Loaded {len(cls.df)} records from CSV")
        else:
            logger.error("Processed data file not found!")
            raise FileNotFoundError("Processed data not found. Run Phase 1 first.")

@app.on_event("startup")
async def startup_event():
    ServiceContainer.load_resources()

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "Zomato Recommendation API"}

@app.get("/api/cities")
async def get_cities():
    """Return list of available cities for autocomplete"""
    if ServiceContainer.df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    cities = sorted(ServiceContainer.df['city'].dropna().unique().tolist())
    return {"cities": cities, "count": len(cities)}

@app.post("/api/recommend", response_model=RecommendationResponse)
async def recommend(request: UserRequest):
    """
    Generate recommendations based on City and Price.
    Integrates Phase 3 (Filter) and Phase 4 (LLM).
    """
    try:
        if ServiceContainer.df is None:
            raise HTTPException(status_code=503, detail="Data not loaded")
            
        # 1. Prepare User Input
        user_input = {"city": request.city, "price": request.price}
        
        # 2. Integration (Phase 3)
        integrator = DataIntegrator(ServiceContainer.df)
        filtered_df = integrator.integrate(user_input)
        
        if len(filtered_df) == 0:
            return {
                "status": "success", 
                "count": 0, 
                "recommendations": [],
                "meta": {"message": "No restaurants found matching criteria"}
            }
            
        # 3. Recommendation (Phase 4)
        engine = GroqEngine()
        # Note: GroqEngine handles the API key internally via os.getenv
        results = engine.generate_recommendations(filtered_df, user_input, top_k=10)
        
        # 4. Format Response
        # Convert DataFrame to list of dicts matching Schema
        recs = results.to_dict(orient='records')
        
        # Normalize fields for schema (e.g. ensure cuisines is list)
        final_recs = []
        for r in recs:
            # Ensure cuisines is a list (it might be a string depending on loading)
            cuisines = r.get('cuisines')
            if isinstance(cuisines, str):
                cuisines = [c.strip() for c in cuisines.split(',')]
            elif cuisines is None:
                cuisines = []
                
            # Ensure dish_liked is a list
            dish_liked = r.get('dish_liked')
            if isinstance(dish_liked, str):
                # Simple heuristic cleanup for string representation of list
                dish_liked = dish_liked.replace('[','').replace(']','').replace("'",'').split(',')
                dish_liked = [d.strip() for d in dish_liked if d.strip()]
            elif dish_liked is None:
                dish_liked = []

            r_clean = {
                "name": r.get('name'),
                "rate": str(r.get('rate')), # Schema expects string
                "votes": int(r.get('votes', 0)),
                "price": int(r.get('price', 0)),
                "location": r.get('location'),
                "cuisines": list(cuisines),
                "rest_type": r.get('rest_type', 'Unknown'),
                "dish_liked": list(dish_liked),
                "reason": r.get('reason'),
                "rank": int(r.get('llm_rank', 0)) if r.get('llm_rank') else None
            }
            final_recs.append(r_clean)
            
        return {
            "status": "success",
            "count": len(final_recs),
            "recommendations": final_recs
        }

    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
