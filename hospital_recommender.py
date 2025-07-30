
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import googlemaps
import os
import logging
import polyline

# Setup logging
logging.basicConfig(filename='hospital_recommender.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Google Maps client
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY")
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

# Default coordinates (Lagos center)
DEFAULT_COORDS = (6.5244, 3.3792)

def get_valid_category(value, default):
    valid_options = {"low", "medium", "high"}
    value = value.strip().lower() if value else default.lower()
    if value in valid_options:
        return value.capitalize()
    logger.warning(f'Invalid input "{value}". Using default: {default}')
    return default.capitalize()

def map_preference_to_value(pref):
    pref_map = {"Low": 0.33, "Medium": 0.66, "High": 1.0}
    return pref_map.get(pref, 0.33)

def compute_service_match(user_service, hospital_services):
    if pd.isna(hospital_services) or pd.isna(user_service):
        logger.warning('Missing service data')
        return 0.0
    user_service = user_service.lower().strip()
    hospital_services = hospital_services.lower().strip()
    hospital_service_list = [s.strip() for s in hospital_services.split(',')]
    if user_service == 'surgery':
        if 'surgery' in hospital_service_list or 'surgical services' in hospital_service_list:
            if all(svc not in hospital_service_list for svc in ['dental surgery', 'oral surgery', 'cosmetic surgery']):
                logger.info(f'Exact match for "surgery" in {hospital_service_list}')
                return 1.0
            else:
                logger.info(f'Excluded mismatch for "surgery" in {hospital_service_list}')
                return 0.0
        elif any('surgery' in svc and 'dental' not in svc and 'oral' not in svc and 'cosmetic' not in svc for svc in hospital_service_list):
            logger.info(f'Partial match for "surgery" in {hospital_service_list}')
            return 0.95
        logger.info(f'No match for "surgery" in {hospital_service_list}')
        return 0.0
    if user_service in hospital_service_list:
        logger.info(f'Exact match for "{user_service}" in {hospital_service_list}')
        return 1.0
    elif any(user_service in svc for svc in hospital_service_list):
        logger.info(f'Strong partial match for "{user_service}" in {hospital_service_list}')
        return 0.95
    elif any(word in ' '.join(hospital_service_list) for word in user_service.split()):
        logger.info(f'Weak partial match for "{user_service}" in {hospital_service_list}')
        return 0.5
    logger.info(f'No match for "{user_service}" in {hospital_service_list}')
    return 0.0

def map_cost_rating(cost_rating):
    if pd.isna(cost_rating) or cost_rating == 'N/A':
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
            return DEFAULT_COORDS, "unknown"
        try:
            lat, lon = map(float, coords_str.strip("()").split(","))
            return (lat, lon), cache.get(f"{address}_city", "unknown")
        except:
            logger.warning(f"Invalid cached coordinates for '{address}'. Re-geocoding.")
    
    try:
        full_address = f"{address}, Lagos, Nigeria"
        geocode_result = gmaps.geocode(full_address)
        if geocode_result:
            location = geocode_result[0]["geometry"]["location"]
            coords = (location["lat"], location["lng"])
            # Extract city from address components
            city = "unknown"
            for component in geocode_result[0]["address_components"]:
                if "locality" in component["types"] or "administrative_area_level_2" in component["types"]:
                    city = component["long_name"].lower()
                    break
            cache[address] = f"({coords[0]},{coords[1]})"
            cache[f"{address}_city"] = city
            return coords, city
        cache[address] = "None"
        cache[f"{address}_city"] = "unknown"
        return DEFAULT_COORDS, "unknown"
    except Exception as e:
        logger.error(f"Geocoding error for '{address}': {e}")
        cache[address] = "None"
        cache[f"{address}_city"] = "unknown"
        return DEFAULT_COORDS, "unknown"

def get_driving_route(user_coords, hospital_coords, hospital_name):
    if user_coords == DEFAULT_COORDS or hospital_coords == DEFAULT_COORDS:
        return None, None, None
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
            instructions = [step["html_instructions"] for step in route["steps"]]
            return distance, duration, "; ".join(instructions)
        return None, None, None
    except Exception as e:
        logger.error(f"Error fetching route to {hospital_name}: {e}")
        return None, None, None

def setup_fuzzy_system():
    cost = ctrl.Antecedent(np.arange(1, 3.1, 0.1), 'cost')
    quality = ctrl.Antecedent(np.arange(2, 5.1, 0.1), 'quality')
    service_match = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'service_match')
    location_match = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'location_match')
    recommendation = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'recommendation')

    # Membership functions
    cost['low'] = fuzz.trimf(cost.universe, [1, 1, 1.5])
    cost['medium'] = fuzz.trimf(cost.universe, [1.2, 1.5, 2])
    cost['high'] = fuzz.trimf(cost.universe, [1.5, 2, 2.5])
    cost['premium'] = fuzz.trimf(cost.universe, [2, 3, 3])

    quality['low'] = fuzz.trimf(quality.universe, [2, 2, 3])
    quality['medium'] = fuzz.trimf(quality.universe, [2.5, 3, 4])
    quality['high'] = fuzz.trimf(quality.universe, [3.5, 4.5, 5])

    service_match['low'] = fuzz.trimf(service_match.universe, [0, 0, 0.3])
    service_match['medium'] = fuzz.trimf(service_match.universe, [0.2, 0.5, 0.7])
    service_match['high'] = fuzz.trimf(service_match.universe, [0.6, 1, 1])

    location_match['low'] = fuzz.trimf(location_match.universe, [0, 0, 0.5])
    location_match['medium'] = fuzz.trimf(location_match.universe, [0.3, 0.5, 0.7])
    location_match['high'] = fuzz.trimf(location_match.universe, [0.5, 1, 1])

    recommendation['low'] = fuzz.trimf(recommendation.universe, [0, 0, 0.4])
    recommendation['medium'] = fuzz.trimf(recommendation.universe, [0.3, 0.5, 0.6])
    recommendation['high'] = fuzz.trimf(recommendation.universe, [0.7, 0.85, 1])

    # Fuzzy rules
    rules = [
        # Low cost preference
        ctrl.Rule(cost['low'] & service_match['high'] & location_match['high'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['low'] & service_match['high'] & location_match['high'] & quality['medium'], recommendation['high']),
        ctrl.Rule(cost['low'] & service_match['high'] & location_match['medium'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['low'] & service_match['medium'] & location_match['high'] & quality['high'], recommendation['medium']),
        ctrl.Rule(cost['low'] & service_match['medium'] & location_match['medium'] & quality['medium'], recommendation['medium']),
        ctrl.Rule(cost['low'] & (service_match['low'] | location_match['low']) & quality['high'], recommendation['low']),
        ctrl.Rule(cost['low'] & service_match['low'] & location_match['low'] & quality['low'], recommendation['low']),
        ctrl.Rule(cost['medium'] & service_match['high'] & location_match['high'] & quality['high'], recommendation['medium']),

        # Medium cost preference
        ctrl.Rule(cost['medium'] & service_match['high'] & location_match['high'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['medium'] & service_match['high'] & location_match['high'] & quality['medium'], recommendation['high']),
        ctrl.Rule(cost['medium'] & service_match['high'] & location_match['medium'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['medium'] & service_match['medium'] & location_match['high'] & quality['high'], recommendation['medium']),
        ctrl.Rule(cost['medium'] & service_match['medium'] & location_match['medium'] & quality['medium'], recommendation['medium']),
        ctrl.Rule(cost['medium'] & (service_match['low'] | location_match['low']) & quality['high'], recommendation['low']),
        ctrl.Rule(cost['medium'] & service_match['low'] & location_match['low'] & quality['low'], recommendation['low']),
        ctrl.Rule(cost['high'] & service_match['high'] & location_match['high'] & quality['high'], recommendation['medium']),

        # High cost preference
        ctrl.Rule(cost['high'] & service_match['high'] & location_match['high'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['high'] & service_match['high'] & location_match['high'] & quality['medium'], recommendation['high']),
        ctrl.Rule(cost['high'] & service_match['high'] & location_match['medium'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['high'] & service_match['medium'] & location_match['high'] & quality['high'], recommendation['medium']),
        ctrl.Rule(cost['high'] & service_match['medium'] & location_match['medium'] & quality['medium'], recommendation['medium']),
        ctrl.Rule(cost['high'] & (service_match['low'] | location_match['low']) & quality['high'], recommendation['low']),
        ctrl.Rule(cost['high'] & service_match['low'] & location_match['low'] & quality['low'], recommendation['low']),

        # Premium cost preference
        ctrl.Rule(cost['premium'] & service_match['high'] & location_match['high'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['premium'] & service_match['high'] & location_match['high'] & quality['medium'], recommendation['high']),
        ctrl.Rule(cost['premium'] & service_match['high'] & location_match['medium'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['premium'] & service_match['medium'] & location_match['high'] & quality['high'], recommendation['medium']),
        ctrl.Rule(cost['premium'] & service_match['medium'] & location_match['medium'] & quality['medium'], recommendation['medium']),
        ctrl.Rule(cost['premium'] & (service_match['low'] | location_match['low']) & quality['high'], recommendation['low']),
        ctrl.Rule(cost['premium'] & service_match['low'] & location_match['low'] & quality['low'], recommendation['low'])
    ]

    recommender_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(recommender_ctrl)

def compute_recommendation_score(row, user_service, fuzzy_system):
    try:
        service_score = compute_service_match(user_service, row["Services"])
        cost_value = map_cost_rating(row["Cost Level"])
        quality_value = float(row["Quality Score"]) if pd.notna(row["Quality Score"]) else 3.0
        location_score = row["Location_Match"]

        fuzzy_system.input["cost"] = cost_value
        fuzzy_system.input["quality"] = quality_value
        fuzzy_system.input["service_match"] = service_score
        fuzzy_system.input["location_match"] = location_score

        fuzzy_system.compute()
        score = fuzzy_system.output.get("recommendation", 0.0)
        logger.info(f"Hospital: {row['Name']}, Service Match: {service_score:.2f}, Location Match: {location_score:.2f}, Score: {score:.3f}")
        return score
    except Exception as e:
        logger.error(f"Error processing {row['Name']}: {e}")
        return 0.0

def recommend_hospitals(location, user_service, cost_pref_str, quality_pref_str):
    """
    Generate hospital recommendations based on user inputs.
    
    Args:
        location (str): Preferred city or address (e.g., 'Ikorodu')
        user_service (str): Desired service (e.g., 'Surgery')
        cost_pref_str (str): Cost preference ('Low', 'Medium', 'High')
        quality_pref_str (str): Quality preference ('Low', 'Medium', 'High')
    
    Returns:
        pd.DataFrame: Recommended hospitals with routing data
    """
    try:
        logger.info("Loading dataset Lagos_hospital.csv")
        dataset_path = "Lagos_hospital.csv"
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset not found at {dataset_path}")
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        data = pd.read_csv(dataset_path)
        data = data.dropna(subset=["Name", "Services", "Cost Level", "Quality Score", "User Rating"])
        data["Full Address"] = data["Full Address"].fillna("Unknown")
        data["Quality Score"] = pd.to_numeric(data["Quality Score"], errors="coerce").fillna(3.0)
        data["User Rating"] = pd.to_numeric(data["User Rating"], errors="coerce").fillna(3.0)

        cost_pref_str = get_valid_category(cost_pref_str, "Medium")
        quality_pref_str = get_valid_category(quality_pref_str, "High")

        cache_file = "hospital_coordinates.csv"
        geocode_cache = load_geocode_cache(cache_file)

        # Geocode user location
        user_coords, user_city = geocode_address(location, geocode_cache)
        if user_city == "unknown":
            logger.warning(f"Could not determine city for location '{location}'. Using all hospitals.")
            data["Location_Match"] = 1.0
        else:
            # Geocode hospital addresses and extract cities
            data["Coordinates"] = data["Full Address"].apply(lambda addr: geocode_address(addr, geocode_cache)[0])
            data["City"] = data["Full Address"].apply(lambda addr: geocode_cache.get(f"{addr}_city", "unknown"))
            data["Location_Match"] = data["City"].apply(lambda city: 1.0 if city.lower() == user_city.lower() else 0.0)
            data = data[data["Location_Match"] == 1.0]
            if data.empty:
                logger.warning(f"No hospitals found in city '{user_city}'")
                return pd.DataFrame()

        save_geocode_cache(geocode_cache, cache_file)

        fuzzy_system = setup_fuzzy_system()
        data["Recommendation_Score"] = data.apply(
            lambda row: compute_recommendation_score(row, user_service, fuzzy_system),
            axis=1
        )

        recommendations = data[data["Recommendation_Score"] > 0].copy()
        if recommendations.empty:
            logger.warning(f"No hospitals found matching service '{user_service}'")
            return pd.DataFrame()

        recommendations = recommendations.sort_values(by="Recommendation_Score", ascending=False).head(3)

        # Add routing information
        for idx, row in recommendations.iterrows():
            distance, duration, instructions = get_driving_route(
                user_coords, row["Coordinates"], row["Name"]
            )
            recommendations.at[idx, "Route_Distance"] = distance
            recommendations.at[idx, "Route_Duration"] = duration
            recommendations.at[idx, "Route_Instructions"] = instructions if instructions else "N/A"

        recommendations.to_csv("recommended_hospitals.csv", index=False)
        logger.info("Recommendations saved to recommended_hospitals.csv")

        return recommendations[
            [
                "Name", "Full Address", "Services", "Cost Level", "Quality Score",
                "Recommendation_Score", "Route_Distance", "Route_Duration", "Route_Instructions"
            ]
        ]
    except Exception as e:
        logger.error(f"Error in recommendation: {e}")
        return pd.DataFrame()