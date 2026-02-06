
import pytest
import pandas as pd
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase3.data_integration import DataIntegrator
from phase4.engine import GroqEngine
from phase5.display import CLIDisplay
from phase2.user_input import UserInputValidator

# Load environment variable for Groq API
load_dotenv(Path(__file__).parent.parent.parent / 'phase4' / '.env')

@pytest.mark.skipif(not os.getenv('GROQ_API_KEY'), reason="GROQ_API_KEY not found")
class TestLivePipeline:
    """
    Live Integration Test for all 5 Phases.
    Strictly limited to 2-3 test cases to avoid API rate limits.
    """

    @pytest.fixture
    def real_data(self):
        """Phase 1: Load real processed data"""
        data_path = Path(__file__).parent.parent.parent / 'phase1' / 'data' / 'processed_zomato_data.pkl'
        if not data_path.exists():
            data_path = Path(__file__).parent.parent.parent / 'phase1' / 'data' / 'processed_zomato_data.csv'
        
        if not data_path.exists():
            pytest.fail("Processed data not found. Run Phase 1 first.")
            
        if data_path.suffix == '.pkl':
            return pd.read_pickle(data_path)
        return pd.read_csv(data_path)

    def test_full_flow_bangalore_budget(self, real_data, capsys):
        """
        Test Case 1: Standard flow
        User wants casual dining in Bangalore (Banashankari) with 800 budget.
        """
        # Phase 2: Mock User Input
        user_input = {'city': 'Banashankari', 'price': 800}
        
        # Phase 3: Data Integration
        integrator = DataIntegrator(real_data)
        filtered_df = integrator.integrate(user_input)
        assert len(filtered_df) > 0, "Phase 3 failed: No restaurants found for valid input"

        # Phase 4: Recommendation Engine (Live Call)
        api_key = os.getenv('GROQ_API_KEY')
        engine = GroqEngine(api_key=api_key)
        
        print("\n[Phase 4] calling Groq API for Test Case 1...")
        recommendations = engine.generate_recommendations(filtered_df, user_input, top_k=5)
        
        assert not recommendations.empty, "Phase 4 failed: No recommendations returned"
        assert 'reason' in recommendations.columns
        assert 'llm_rank' in recommendations.columns

        # Phase 5: Display
        CLIDisplay.display_recommendations(recommendations, user_input)
        
        # Capture output to verify Display logic
        captured = capsys.readouterr()
        output = captured.out
        
        assert "RESTAURANT RECOMMENDATIONS" in output
        assert "Banashankari" in output
        assert "Why recommended:" in output
        
    def test_full_flow_high_budget(self, real_data, capsys):
        """
        Test Case 2: High budget flow
        User wants specific location with 2000 budget.
        """
        # Phase 2: Mock User Input
        user_input = {'city': 'Basavanagudi', 'price': 2000}
        
        # Phase 3: Data Integration
        integrator = DataIntegrator(real_data)
        filtered_df = integrator.integrate(user_input)
        assert len(filtered_df) > 0

        # Phase 4: Recommendation Engine (Live Call)
        api_key = os.getenv('GROQ_API_KEY')
        engine = GroqEngine(api_key=api_key)
        
        print("\n[Phase 4] calling Groq API for Test Case 2...")
        recommendations = engine.generate_recommendations(filtered_df, user_input, top_k=3)
        
        assert not recommendations.empty
        
        # Phase 5: Display
        CLIDisplay.display_recommendations(recommendations, user_input)
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "INR 2000" in output
        assert "Why recommended:" in output
