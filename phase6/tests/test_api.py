
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase6.main import app, ServiceContainer

client = TestClient(app)

# Mock Data
MOCK_DF = pd.DataFrame([
    {
        'name': 'Test Restaurant',
        'rate': 4.5,
        'votes': 100,
        'price': 500,
        'location': 'Bangalore',
        'city': 'Bangalore',
        'cuisines': 'North Indian',
        'rest_type': 'Casual Dining',
        'dish_liked': "['Curry']",
        'online_order': 'Yes',
        'book_table': 'No'
    },
    {
        'name': 'Cafe Coffee',
        'rate': 4.0,
        'votes': 50,
        'price': 300,
        'location': 'Indiranagar',
        'city': 'Bangalore',
        'cuisines': 'Cafe',
        'rest_type': 'Cafe',
        'dish_liked': "['Coffee']",
        'online_order': 'Yes',
        'book_table': 'No'
    }
])

@pytest.fixture
def mock_container():
    """Mock the ServiceContainer data for tests"""
    original_df = ServiceContainer.df
    ServiceContainer.df = MOCK_DF
    yield
    ServiceContainer.df = original_df

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_get_cities(mock_container):
    response = client.get("/api/cities")
    assert response.status_code == 200
    data = response.json()
    assert "cities" in data
    assert "Bangalore" in data["cities"]

@patch('phase6.main.GroqEngine')
def test_recommend_endpoint(mock_engine_cls, mock_container):
    # Mock GroqEngine instance and generate_recommendations method
    mock_instance = MagicMock()
    mock_engine_cls.return_value = mock_instance
    
    # Mock return value of generate_recommendations (DataFrame)
    mock_result_df = pd.DataFrame([
        {
            'name': 'Test Restaurant',
            'rate': 4.5,
            'votes': 100,
            'price': 500,
            'location': 'Bangalore',
            'cuisines': 'North Indian',
            'rest_type': 'Casual Dining',
            'dish_liked': "['Curry']",
            'reason': 'Great food',
            'llm_rank': 1
        }
    ])
    mock_instance.generate_recommendations.return_value = mock_result_df

    payload = {"city": "Bangalore", "price": 1000}
    response = client.post("/api/recommend", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["recommendations"]) == 1
    assert data["recommendations"][0]["name"] == "Test Restaurant"
    assert data["recommendations"][0]["rank"] == 1

def test_recommend_endpoint_no_results(mock_container):
    # Price too low, should return empty list from DataIntegrator (Phase 3 logic)
    payload = {"city": "Bangalore", "price": 100}
    response = client.post("/api/recommend", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert len(data["recommendations"]) == 0
