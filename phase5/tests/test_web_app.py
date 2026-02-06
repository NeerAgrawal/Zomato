import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase5.web_app import load_data

class TestPhase3:
    @pytest.fixture
    def mock_data(self):
        """Create mock dataframe"""
        return pd.DataFrame({
            'name': ['Test Rest 1', 'Test Rest 2'],
            'city': ['Banashankari', 'Indiranagar'],
            'rate': ['4.5/5', '3.8/5'],
            'votes': [100, 50],
            'price': [500, 800],
            'location': ['Banashankari', 'Indiranagar'],
            'cuisines': [['Italian'], ['Chinese']]
        })

    def test_load_data_failure(self):
        """Test graceful failure when data is missing"""
        with patch('pandas.read_pickle', side_effect=FileNotFoundError), \
             patch('pandas.read_csv', side_effect=FileNotFoundError), \
             patch('pathlib.Path.exists', return_value=False):
            
            with pytest.raises(FileNotFoundError):
                load_data()

    def test_environment_setup(self):
        """Verify imports work (system path setup)"""
        try:
            import phase5.web_app
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import app module: {e}")
