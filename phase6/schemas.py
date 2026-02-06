from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any

class UserRequest(BaseModel):
    city: str = Field(..., min_length=2, description="City to search in (e.g., 'Bangalore')")
    price: float = Field(..., gt=0, description="Maximum price for two people")
    
    @field_validator('city')
    def validate_city(cls, v):
        if not v.strip():
            raise ValueError("City cannot be empty")
        return v.title()

class RestaurantDetail(BaseModel):
    name: str
    rate: str
    votes: int
    price: int
    location: str
    cuisines: List[str]
    rest_type: str
    dish_liked: List[str]
    reason: Optional[str] = None
    rank: Optional[int] = None
    
    # Allow extra fields for flexibility
    model_config = {"extra": "ignore"}

class RecommendationResponse(BaseModel):
    status: str
    count: int
    recommendations: List[RestaurantDetail]
    meta: Optional[Dict[str, Any]] = None
