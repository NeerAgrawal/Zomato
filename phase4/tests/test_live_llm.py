import pytest
import pandas as pd
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase4.engine import GroqEngine

# Load phase4 env explicitly
load_dotenv(Path(__file__).parent.parent / '.env')

@pytest.mark.skipif(not os.getenv('GROQ_API_KEY'), reason="GROQ_API_KEY not found")
class TestLiveGroqEngine:
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'name': ['Truffles', 'Empire', 'Meghana Foods'],
            'rate': [4.5, 4.2, 4.4],
            'votes': [5000, 4000, 6000],
            'price': [800, 700, 900],
            'cuisines': [['Burger', 'Cafe'], ['North Indian', 'Mughlai'], ['Biryani', 'Andhra']],
            'rest_type': ['Cafe', 'Casual Dining', 'Casual Dining']
        })

    def test_live_recommendation(self, sample_data):
        """test that actual API call returns valid recommendations with reasoning"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key or "your_groq_api_key" in api_key:
            pytest.skip("Skipping live test: Invalid or missing API Key")
            
        print(f"\nTesting with API Key: {api_key[:5]}...{api_key[-4:]}")
        engine = GroqEngine(api_key=api_key)
        
        user_input = {'city': 'Bangalore', 'price': 1000}
        
        results = engine.generate_recommendations(sample_data, user_input, top_k=2)
        
        # Verify results
        assert not results.empty, "Result should not be empty"
        assert 'llm_rank' in results.columns, "Output should contain 'llm_rank' which indicates LLM usage"
        assert 'reason' in results.columns
        
        # Verify reasoning is not the fallback message
        fallback_msg = "Best match based on rating and popularity."
        # The LLM generating distinct reasons is a proof of life
        reasons = results['reason'].tolist()
        print("\nGenerated Reasons:")
        for r in reasons:
            print(f"- {r}")
            
        assert any(r != fallback_msg for r in reasons), "Reasons should differ from fallback message"

    def test_live_parsing_resilience(self, sample_data):
        """Test engine's ability to handle the response parsing from a real call"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            pytest.skip("No API Key")
            
        engine = GroqEngine(api_key=api_key)
        user_input = {'city': 'Bangalore', 'price': 1000}
        results = engine.generate_recommendations(sample_data, user_input, top_k=2)
        assert len(results) <= 2

    def test_live_unique_reasons(self, sample_data):
        """Test that LLM provides unique reasons for each recommendation"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            pytest.skip("No API Key")
            
        engine = GroqEngine(api_key=api_key)
        user_input = {'city': 'Bangalore', 'price': 1000}
        
        # Request more items to increase chance of duplication if logic is bad
        results = engine.generate_recommendations(sample_data, user_input, top_k=3)
        
        reasons = results['reason'].tolist()
        print("\nVerifying Reasons:")
        for i, r in enumerate(reasons):
            print(f"{i+1}. {r}")
            
        # Check uniqueness
        unique_reasons = set(reasons)
        assert len(unique_reasons) == len(reasons), f"Found duplicate reasons! {len(reasons)} total vs {len(unique_reasons)} unique."
        
        # Check specificity (simple heuristic: length > 15 chars)
        for r in reasons:
            assert len(r) > 15, f"Reason is too short/generic: '{r}'"
