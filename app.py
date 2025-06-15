from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from hospital_recommender import recommend_hospitals
import pandas as pd
import os

app = FastAPI(title="Hospital Recommender API")

# Input model for the recommendation request
class RecommendationRequest(BaseModel):
    location: str
    service_needed: str
    cost_preference: str  # Low, Medium, High
    quality_preference: str  # Low, Medium, High

# Output model for the recommendation response
class RecommendationResponse(BaseModel):
    name: str
    full_address: str
    services: str
    cost_level: str
    quality_score: float
    recommendation_score: float
    route_distance: str | None
    route_duration: str | None
    route_instructions: str | None

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=list[RecommendationResponse])
async def get_recommendations(request: RecommendationRequest):
    try:
        # Validate categorical inputs
        valid_categories = {"Low", "Medium", "High"}
        if request.cost_preference.capitalize() not in valid_categories:
            raise HTTPException(status_code=400, detail="Invalid cost preference. Must be Low, Medium, or High.")
        if request.quality_preference.capitalize() not in valid_categories:
            raise HTTPException(status_code=400, detail="Invalid quality preference. Must be Low, Medium, or High.")
        
        # Load dataset
        dataset_path = "Lagos_hospital.csv"
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=500, detail="Hospital dataset not found.")
        
        # Call recommendation logic
        recommendations = recommend_hospitals(
            location=request.location,
            user_service=request.service_needed,
            cost_pref_str=request.cost_preference.capitalize(),
            quality_pref_str=request.quality_preference.capitalize()
        )
        
        if recommendations.empty:
            raise HTTPException(status_code=404, detail="No hospitals found matching your criteria.")
        
        # Convert recommendations to response model
        response = [
            RecommendationResponse(
                name=row["Name"],
                full_address=row["Full Address"],
                services=row["Services"],
                cost_level=row["Cost Level"],
                quality_score=row["Quality Score"],
                recommendation_score=row["Recommendation_Score"],
                route_distance=row.get("Route_Distance"),
                route_duration=row.get("Route_Duration"),
                route_instructions=row.get("Route_Instructions")
            )
            for _, row in recommendations.iterrows()
        ]
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")