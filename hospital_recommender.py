import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from geopy.distance import geodesic
import googlemaps
import os
import polyline
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Google Maps client
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY")
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

DEFAULT_COORDS = (6.5244, 3.3792)

def get_valid_category(value, default):
    valid_options = {"low", "medium", "high"}
    value = value.strip().lower() if value else default.lower()
    if value in valid_options:
        return value.capitalize()
    return default.capitalize()

def map_preference_to_value(pref):
    pref_map = {"Low": 0.33, "Medium": 0.66, "High": 1.0}
    return pref_map.get(pref, 0.33)

def compute_service_match(user_service, hospital_services):
    if pd.isna(hospital_services) or pd.isna(user_service):
        return 0.0
    user_service = user_service.lower().strip()
    hospital_services = hospital_services.lower().strip()
    if user_service in hospital_services:
        return 1.0
    elif any(word in hospital_services for word in user_service.split()):
        return 0.5
    return 0.0

def map_cost_rating(cost_rating):
    if pd.isna(cost_rating):
        return 1.0
    cost_map = {"Low": 1.0, "Medium": 2.0, "High": 3.0, "Premium": 3.0}
    return cost_map.get(cost_rating.strip().capitalize(), 1.0)

def load_geocode_cache(cache_file="hospital_coordinates.csv"):
    if os.path.exists(cache_file):
        try:
            cache = pd.read_csv(cache_file, index_col="Address")
            return cache.to_dict()["Coordinates"]
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    return {}

def save_geocode_cache(cache, cache_file="hospital_coordinates.csv"):
    try:
        cache_df = pd.DataFrame.from_dict(cache, orient="index", columns=["Coordinates"])
        cache_df.index.name = "Address"
        cache_df.to_csv(cache_file)
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

def geocode_address(address, cache):
    if address in cache:
        coords_str = cache[address]
        if coords_str == "None":
            return DEFAULT_COORDS
        try:
            lat, lon = map(float, coords_str.strip("()").split(","))
            return (lat, lon)
        except:
            logger.warning(f"Invalid cached coordinates for '{address}'. Re-geocoding.")
    
    try:
        full_address = f"{address}, Lagos, Nigeria"
        geocode_result = gmaps.geocode(full_address)
        if geocode_result:
            location = geocode_result[0]["geometry"]["location"]
            coords = (location["lat"], location["lng"])
            cache[address] = f"({coords[0]},{coords[1]})"
            return coords
        cache[address] = "None"
        return DEFAULT_COORDS
    except Exception as e:
        logger.error(f"Geocoding error for '{address}': {e}")
        cache[address] = "None"
        return DEFAULT_COORDS

def calculate_distance(user_coords, hospital_coords, max_distance=10.0):
    if user_coords is None or hospital_coords is None:
        return 0.0, float("inf")
    try:
        distance = geodesic(user_coords, hospital_coords).km
        normalized = max(0.0, 1.0 - (distance / max_distance))
        return normalized, distance
    except Exception as e:
        logger.error(f"Distance calculation error: {e}")
        return 0.0, float("inf")

def get_driving_route(user_coords, hospital_coords, hospital_name):
    if user_coords == DEFAULT_COORDS or hospital_coords == DEFAULT_COORDS:
        return None, None, None, None
    try:
        directions_result = gmaps.directions(
            origin=user_coords,
            destination=hospital_coords,
            mode="driving",
            departure_time=datetime.now()
        )
        if directions_result and len(directions_result) > 0:
            route = directions_result[0]["legs"][0]
            distance = route["distance"]["text"]
            duration = route["duration"]["text"]
            polyline_points = route["overview_polyline"]["points"]
            instructions = [step["html_instructions"] for step in route["steps"]]
            return distance, duration, polyline_points, instructions
        return None, None, None, None
    except Exception as e:
        logger.error(f"Error fetching route to {hospital_name}: {e}")
        return None, None, None, None



def setup_fuzzy_system():

    # Membership function definition
    cost = ctrl.Antecedent(np.arange(1, 3.1, 0.1), "cost")
    quality = ctrl.Antecedent(np.arange(3, 5.1, 0.1), "quality")
    user_rating = ctrl.Antecedent(np.arange(1, 5.1, 0.1), "user_rating")
    service_match = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "service_match")
    user_cost_pref = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "user_cost_pref")
    user_quality_pref = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "user_quality_pref")
    proximity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "proximity")
    recommendation = ctrl.Consequent(np.arange(0, 1.1, 0.1), "recommendation")

    cost["low"] = fuzz.trapmf(cost.universe, [1, 1, 1.2, 1.8])
    cost["medium"] = fuzz.trapmf(cost.universe, [1.2, 1.8, 2.2, 2.8])
    cost["high"] = fuzz.trapmf(cost.universe, [2.2, 2.8, 3, 3])

    quality["low"] = fuzz.trapmf(quality.universe, [3, 3, 3.4, 3.8])
    quality["medium"] = fuzz.trapmf(quality.universe, [3.4, 3.8, 4.2, 4.6])
    quality["high"] = fuzz.trapmf(quality.universe, [4.2, 4.6, 5, 5])

    user_rating["low"] = fuzz.trapmf(user_rating.universe, [1, 1, 2, 3])
    user_rating["medium"] = fuzz.trapmf(user_rating.universe, [2, 3, 3.5, 4])
    user_rating["high"] = fuzz.trapmf(user_rating.universe, [3.5, 4, 5, 5])

    service_match["low"] = fuzz.trapmf(service_match.universe, [0, 0, 0.3, 0.6])
    service_match["high"] = fuzz.trapmf(service_match.universe, [0.4, 0.7, 1, 1])

    user_cost_pref["low"] = fuzz.trapmf(user_cost_pref.universe, [0, 0, 0.2, 0.4])
    user_cost_pref["medium"] = fuzz.trapmf(user_cost_pref.universe, [0.3, 0.5, 0.7, 0.9])
    user_cost_pref["high"] = fuzz.trapmf(user_cost_pref.universe, [0.6, 0.8, 1, 1])

    user_quality_pref["low"] = fuzz.trapmf(user_quality_pref.universe, [0, 0, 0.2, 0.4])
    user_quality_pref["medium"] = fuzz.trapmf(user_quality_pref.universe, [0.3, 0.5, 0.7, 0.9])
    user_quality_pref["high"] = fuzz.trapmf(user_quality_pref.universe, [0.6, 0.8, 1, 1])

    proximity["far"] = fuzz.trapmf(proximity.universe, [0, 0, 0.2, 0.4])
    proximity["medium"] = fuzz.trapmf(proximity.universe, [0.3, 0.4, 0.6, 0.7])
    proximity["near"] = fuzz.trapmf(proximity.universe, [0.6, 0.7, 0.9, 1])
    proximity["very_near"] = fuzz.trapmf(proximity.universe, [0.8, 0.9, 1, 1])

    recommendation["low"] = fuzz.trapmf(recommendation.universe, [0, 0, 0.3, 0.5])
    recommendation["medium"] = fuzz.trapmf(recommendation.universe, [0.4, 0.5, 0.6, 0.7])
    recommendation["high"] = fuzz.trapmf(recommendation.universe, [0.6, 0.7, 1, 1])


# Fuzzy rules
    rules = [
        ctrl.Rule(
            service_match["high"] & proximity["very_near"] & quality["high"] & user_rating["high"] &
            ((cost["low"] & user_cost_pref["low"]) | (cost["medium"] & user_cost_pref["medium"]) | (cost["high"] & user_cost_pref["high"])) &
            user_quality_pref["high"],
            recommendation["high"]
        ),
        ctrl.Rule(
            service_match["high"] & proximity["near"] & quality["high"] & user_rating["medium"] &
            ((cost["low"] & user_cost_pref["low"]) | (cost["medium"] & user_cost_pref["medium"])) &
            user_quality_pref["high"],
            recommendation["high"]
        ),
        ctrl.Rule(
            service_match["high"] & proximity["medium"] & (quality["medium"] | quality["high"]) & user_rating["medium"] &
            ((cost["low"] & user_cost_pref["low"]) | (cost["medium"] & user_cost_pref["medium"])),
            recommendation["medium"]
        ),
        ctrl.Rule(
            service_match["high"] & proximity["near"] & quality["medium"] & user_rating["medium"] &
            user_quality_pref["medium"],
            recommendation["medium"]
        ),
        ctrl.Rule(
             service_match["low"] | proximity["far"] | (quality["low"] & user_quality_pref["high"]),
            recommendation["low"]
        ),
        ctrl.Rule(
            (cost["high"] & user_cost_pref["low"]) | (cost["medium"] & user_cost_pref["low"]),
            recommendation["low"]
        ),
        ctrl.Rule(
            service_match["high"] & proximity["very_near"] & quality["medium"] & user_rating["high"] &
            user_quality_pref["medium"] & (cost["low"] | cost["medium"]),
            recommendation["high"]
        ),
        ctrl.Rule(
            service_match["low"] & proximity["very_near"] & quality["high"] & user_rating["high"] &
            user_quality_pref["high"] & cost["low"] & user_cost_pref["low"],
            recommendation["medium"]
        )
    ]

    hospital_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(hospital_ctrl)


# Inference system 
def compute_recommendation_score(row, user_service, user_cost_pref, user_quality_pref, user_coords, fuzzy_system):
    try:
        service_score = compute_service_match(user_service, row["Services"])
        cost_value = map_cost_rating(row["Cost Level"])
        quality_value = float(row["Quality Score"]) if pd.notna(row["Quality Score"]) else 3.1
        user_rating_value = float(row["User Rating"]) if pd.notna(row["User Rating"]) else 3.0
        proximity_score, _ = calculate_distance(user_coords, row.get("Coordinates"))


# Fuzzification process
        fuzzy_system.input["cost"] = cost_value
        fuzzy_system.input["quality"] = quality_value
        fuzzy_system.input["user_rating"] = user_rating_value
        fuzzy_system.input["service_match"] = service_score
        fuzzy_system.input["user_cost_pref"] = user_cost_pref
        fuzzy_system.input["user_quality_pref"] = user_quality_pref
        fuzzy_system.input["proximity"] = proximity_score

        fuzzy_system.compute()
        return fuzzy_system.output.get("recommendation", 0.0)
    except Exception as e:
        logger.error(f"Error processing {row['Name']}: {e}")
        return 0.0

def recommend_hospitals(location, user_service, cost_pref_str, quality_pref_str):
    try:
        logger.info("Loading dataset Lagos_hospital.csv")
        data = pd.read_csv("Lagos_hospital.csv")
        data = data.dropna(subset=["Name", "Services", "Cost Level", "Quality Score", "User Rating"])
        data["Full Address"] = data["Full Address"].fillna("Unknown")
        data["Quality Score"] = pd.to_numeric(data["Quality Score"], errors="coerce").fillna(3.1)
        data["User Rating"] = pd.to_numeric(data["User Rating"], errors="coerce").fillna(3.0)

        cost_pref = map_preference_to_value(cost_pref_str)
        quality_pref = map_preference_to_value(quality_pref_str)

        cache_file = "hospital_coordinates.csv"
        geocode_cache = load_geocode_cache(cache_file)

        user_coords = None
        if location:
            user_coords = geocode_address(location, geocode_cache)
            if user_coords == DEFAULT_COORDS:
                logger.warning(f"Could not geocode location '{location}'. Using all hospitals without distance filter.")
        
        data["Coordinates"] = data["Full Address"].apply(lambda addr: geocode_address(addr, geocode_cache))
        save_geocode_cache(geocode_cache, cache_file)

        if user_coords and user_coords != DEFAULT_COORDS:
            data["Distance_km"] = data["Coordinates"].apply(lambda coords: calculate_distance(user_coords, coords)[1])
            data = data[data["Distance_km"] <= 10.0]

        if data.empty:
            logger.warning("No hospitals available after filtering")
            return pd.DataFrame()

        fuzzy_system = setup_fuzzy_system()
        data["Recommendation_Score"] = data.apply(
            lambda row: compute_recommendation_score(row, user_service, cost_pref, quality_pref, user_coords, fuzzy_system),
            axis=1
        )

        recommendations = data[data["Recommendation_Score"] > 0].copy()
        if recommendations.empty:
            logger.warning(f"No hospitals found matching service '{user_service}'")
            return pd.DataFrame()

        recommendations = recommendations.sort_values(by="Recommendation_Score", ascending=False).head(3)

        for idx, row in recommendations.iterrows():
            distance, duration, polyline_points, instructions = get_driving_route(
                user_coords, row["Coordinates"], row["Name"]
            )
            recommendations.at[idx, "Route_Distance"] = distance
            recommendations.at[idx, "Route_Duration"] = duration
            recommendations.at[idx, "Polyline_Points"] = polyline_points
            recommendations.at[idx, "Route_Instructions"] = "; ".join(instructions) if instructions else "N/A"

        return recommendations[
            [
                "Name", "Full Address", "Services", "Cost Level", "Quality Score",
                "Recommendation_Score", "Coordinates", "Route_Distance", "Route_Duration", "Route_Instructions"
            ]
        ]
    except Exception as e:
        logger.error(f"Error in recommendation: {e}")
        return pd.DataFrame()
