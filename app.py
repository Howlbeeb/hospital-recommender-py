from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from hospital_recommender import recommend_hospitals
import pandas as pd
import os

app = FastAPI(title="Hospital Recommender API")

class RecommendationRequest(BaseModel):
    location: str
    service_needed: str
    cost_preference: str
    quality_preference: str

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

@app.get("/", response_class=RedirectResponse)
async def root():
    return "/docs"

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=list[RecommendationResponse])
async def get_recommendations(request: RecommendationRequest):
    try:
        valid_categories = {"Low", "Medium", "High"}
        cost_pref = request.cost_preference.capitalize()
        quality_pref = request.quality_preference.capitalize()
        if cost_pref not in valid_categories:
            raise HTTPException(status_code=400, detail="Invalid cost preference. Must be Low, Medium, or High.")
        if quality_pref not in valid_categories:
            raise HTTPException(status_code=400, detail="Invalid quality preference. Must be Low, Medium, or High.")
          
        dataset_path = "Lagos_hospital.csv"
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=500, detail="Hospital dataset not found. Please ensure Lagos_hospital.csv is available.")
          
        try:
            df = pd.read_csv(dataset_path)
            required_columns = ["Name", "Full Address", "Services", "Cost Level", "Quality Score", "User Rating"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise HTTPException(status_code=500, detail=f"Dataset missing required columns: {missing_columns}")
        except pd.errors.ParserError:
            raise HTTPException(status_code=500, detail="Invalid CSV format in Lagos_hospital.csv. Check for correct headers and delimiters.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")
          
        recommendations = recommend_hospitals(
            location=request.location,
            user_service=request.service_needed,
            cost_pref_str=cost_pref,
            quality_pref_str=quality_pref
          )
          
        if recommendations.empty:
            raise HTTPException(status_code=404, detail="No hospitals found matching your criteria.")
          
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
