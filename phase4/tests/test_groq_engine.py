import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase4.engine import GroqEngine
from phase4.prompts import PromptDesign
from phase4.parser import ResponseParser
from phase4.context import ContextManager

class TestGroqEngine:
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'name': ['Pizza Hut', 'Dominos', 'Burger King'],
            'rate': [4.1, 4.0, 3.8],
            'votes': [100, 200, 50],
            'price': [500, 400, 300],
            'cuisines': [['Pizza'], ['Pizza'], ['Burger']],
            'rest_type': ['Casual', 'Casual', 'Fast Food']
        })

    def test_context_manager(self, sample_data):
        """Test if context string is generated correctly"""
        context = ContextManager.prepare_context(sample_data)
        assert "Pizza Hut" in context
        assert "Dominos" in context
        assert "Rating: 4.1" in context

    def test_parser_valid_json(self):
        """Test parsing of valid JSON response"""
        json_str = '[{"name": "Pizza Hut", "rank": 1, "reason": "Good"}]'
        parsed = ResponseParser.parse_response(json_str)
        assert len(parsed) == 1
        assert parsed[0]['name'] == "Pizza Hut"

    def test_parser_markdown_json(self):
        """Test parsing of JSON inside markdown blocks"""
        md_str = '```json\n[{"name": "Pizza Hut", "rank": 1, "reason": "Good"}]\n```'
        parsed = ResponseParser.parse_response(md_str)
        assert len(parsed) == 1
        assert parsed[0]['name'] == "Pizza Hut"

    def test_engine_fallback(self, sample_data):
        """Test fallback mechanism when API key is missing"""
        engine = GroqEngine(api_key=None)
        # Mocking empty client to ensure fallback
        
        results = engine.generate_recommendations(sample_data, {'city': 'Test', 'price': 1000})
        assert len(results) == 3
        # Should be sorted by rate desc (Pizza Hut 4.1 first)
        assert results.iloc[0]['name'] == 'Pizza Hut'
        assert "Best match based on rating" in results.iloc[0]['reason']

    @patch('phase4.engine.Groq')
    def test_engine_llm_flow(self, mock_groq_class, sample_data):
        """Test full flow with mocked LLM response"""
        # Setup mock client instance
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        
        # Setup mock completion response
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = json.dumps([
            {"name": "Dominos", "rank": 1, "reason": "High votes"},
            {"name": "Pizza Hut", "rank": 2, "reason": "Good rating"}
        ])
        mock_client.chat.completions.create.return_value = mock_completion
        
        # instantiate engine - this should use the mocked Groq class
        engine = GroqEngine(api_key="fake-key")
        
        results = engine.generate_recommendations(sample_data, {'city': 'Test', 'price': 1000})
        
        assert len(results) == 2
        # Dominos should be first because LLM ranked it 1
        assert results.iloc[0]['name'] == 'Dominos'
        assert results.iloc[0]['reason'] == "High votes"
        assert results.iloc[1]['name'] == 'Pizza Hut'
