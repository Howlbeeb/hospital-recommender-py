# recommender.py

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import logging
import re
import os
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from functools import lru_cache

# Setup logging
logging.basicConfig(filename='hospital_recommender.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Initialize geocoder
geolocator = Nominatim(user_agent="hospital_recommender")

# Cache geocoding results
@lru_cache(maxsize=1000)
def geocode_address(address):
    try:
        location = geolocator.geocode(address + ", Lagos, Nigeria", timeout=10)
        if location:
            return (location.latitude, location.longitude)
        logger.warning(f"Geocoding failed for {address}")
        return None
    except Exception as e:
        logger.error(f"Geocoding error for {address}: {str(e)}")
        return None

# Utility functions
def get_valid_category(value, default):
    valid_options = {'low', 'medium', 'high'}
    value = value.strip().lower() if value else default.lower()
    if value in valid_options:
        return value.capitalize()
    logger.warning(f'Invalid input "{value}". Using default: {default}')
    return default.capitalize()

def map_preference_to_value(pref):
    pref_map = {'Low': 0.2, 'Medium': 0.5, 'High': 0.8}
    return pref_map.get(pref, 0.2)

def compute_service_match(user_service, hospital_services, valid_services):
    if pd.isna(hospital_services) or pd.isna(user_service):
        logger.warning('Missing service data')
        return 0.0
    user_service = user_service.lower().strip()
    hospital_services = [s.strip().lower() for s in hospital_services.split(',')]
    if user_service == 'surgery':
        if 'surgery' in hospital_services or 'surgical services' in hospital_services:
            if all(svc not in hospital_services for svc in ['dental surgery', 'oral surgery', 'cosmetic surgery']):
                logger.info(f'Exact match for "surgery" in {hospital_services}')
                return 1.0
            else:
                logger.info(f'Excluded mismatch for "surgery" in {hospital_services}')
                return 0.0
        elif any('surgery' in svc and 'dental' not in svc and 'oral' not in svc and 'cosmetic' not in svc for svc in hospital_services):
            logger.info(f'Partial match for "surgery" in {hospital_services}')
            return 0.95
        logger.info(f'No match for "surgery" in {hospital_services}')
        return 0.0
    if user_service in hospital_services:
        logger.info(f'Exact match for "{user_service}" in {hospital_services}')
        return 1.0
    elif any(user_service in svc for svc in hospital_services):
        logger.info(f'Strong partial match for "{user_service}" in {hospital_services}')
        return 0.95
    elif any(word in ' '.join(hospital_services) for word in user_service.split()):
        logger.info(f'Weak partial match for "{user_service}" in {hospital_services}')
        return 0.3
    logger.info(f'No match for "{user_service}" in {hospital_services}')
    return 0.0

def map_cost_rating(cost_rating):
    if pd.isna(cost_rating) or cost_rating == 'N/A':
        return 1.5
    cost_map = {'Low': 1.0, 'Medium': 1.5, 'High': 2.0, 'Premium': 3.0}
    return cost_map.get(cost_rating.strip().capitalize(), 1.5)

def compute_distance_match(user_coords, hospital_coords, max_distance=10.0):
    if not user_coords or not hospital_coords:
        logger.warning('Missing coordinates for distance calculation')
        return 0.0
    try:
        distance = geodesic(user_coords, hospital_coords).km
        # Normalize distance to 0–1 (closer = higher score)
        distance_match = max(0.0, 1.0 - (distance / max_distance))
        logger.info(f"Distance: {distance:.2f} km, Distance Match: {distance_match:.2f}")
        return distance_match
    except Exception as e:
        logger.error(f"Distance calculation error: {str(e)}")
        return 0.0

# Load and preprocess dataset
def load_and_preprocess_data():
    dataset_path = "Lagos_hospital.csv"
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    df = pd.read_csv(dataset_path)
    df['Cost_Numeric'] = df['Cost Level'].apply(map_cost_rating)
    df['Quality_Numeric'] = pd.to_numeric(df['Quality Score'], errors='coerce').fillna(3.0)
    df['Services'] = df['Services'].fillna('Unknown')
    df['Full Address'] = df['Full Address'].fillna('Unknown')
    
    # Geocode hospital addresses
    df['Coordinates'] = df['Full Address'].apply(geocode_address)
    df['City'] = df['Full Address'].str.split(',').str[-1].str.strip()  # Fallback city
    df['City'] = df['City'].str.lower().replace('', 'unknown').fillna('unknown')
    
    # Log geocoding results
    logger.info(f"Geocoded {len(df[df['Coordinates'].notnull()])}/{len(df)} addresses")
    logger.info(f'Extracted cities: {df["City"].unique()}')
    
    # Dynamic valid services
    valid_services = set()
    for services in df['Services'].dropna():
        valid_services.update([s.strip().lower() for s in services.split(',')])
    valid_services = list(valid_services)
    logger.info(f'Valid services: {valid_services}')
    
    return df, valid_services

# Fuzzy logic system
def setup_fuzzy_system(df):
    quality_min = max(1, df['Quality_Numeric'].min())
    quality_max = min(5, df['Quality_Numeric'].max() + 0.1)
    cost_min = max(1, df['Cost_Numeric'].min())
    cost_max = min(3, df['Cost_Numeric'].max() + 0.1)
    
    cost = ctrl.Antecedent(np.arange(cost_min, cost_max, 0.1), 'cost')
    quality = ctrl.Antecedent(np.arange(quality_min, quality_max, 0.1), 'quality')
    service_match = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'service_match')
    distance_match = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'distance_match')  # Updated
    recommendation = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'recommendation')

    cost['low'] = fuzz.trimf(cost.universe, [cost_min, cost_min, (cost_min + cost_max) / 2])
    cost['medium'] = fuzz.trimf(cost.universe, [cost_min + 0.2, (cost_min + cost_max) / 2, cost_max - 0.5])
    cost['high'] = fuzz.trimf(cost.universe, [(cost_min + cost_max) / 2, cost_max - 0.5, cost_max])
    cost['premium'] = fuzz.trimf(cost.universe, [cost_max - 0.5, cost_max, cost_max])

    quality['low'] = fuzz.trimf(quality.universe, [quality_min, quality_min, quality_min + 1])
    quality['medium'] = fuzz.trimf(quality.universe, [quality_min + 0.5, (quality_min + quality_max) / 2, quality_max - 0.5])
    quality['high'] = fuzz.trimf(quality.universe, [quality_max - 1, quality_max, quality_max])

    service_match['low'] = fuzz.trimf(service_match.universe, [0, 0, 0.3])
    service_match['medium'] = fuzz.trimf(service_match.universe, [0.2, 0.5, 0.7])
    service_match['high'] = fuzz.trimf(service_match.universe, [0.6, 1, 1])

    distance_match['low'] = fuzz.trimf(distance_match.universe, [0, 0, 0.3])
    distance_match['medium'] = fuzz.trimf(distance_match.universe, [0.2, 0.5, 0.7])
    distance_match['high'] = fuzz.trimf(distance_match.universe, [0.6, 1, 1])

    recommendation['low'] = fuzz.trimf(recommendation.universe, [0, 0, 0.4])
    recommendation['medium'] = fuzz.trimf(recommendation.universe, [0.3, 0.5, 0.6])
    recommendation['high'] = fuzz.trimf(recommendation.universe, [0.7, 0.85, 1])

    rules = [
        # Low cost preference
        ctrl.Rule(cost['low'] & service_match['high'] & distance_match['high'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['low'] & service_match['high'] & distance_match['high'] & quality['medium'], recommendation['high']),
        ctrl.Rule(cost['low'] & service_match['high'] & distance_match['medium'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['low'] & service_match['medium'] & distance_match['high'] & quality['high'], recommendation['medium']),
        ctrl.Rule(cost['low'] & service_match['medium'] & distance_match['medium'] & quality['medium'], recommendation['medium']),
        ctrl.Rule(cost['low'] & (service_match['low'] | distance_match['low']) & quality['high'], recommendation['low']),
        ctrl.Rule(cost['low'] & service_match['low'] & distance_match['low'] & quality['low'], recommendation['low']),
        ctrl.Rule(cost['medium'] & service_match['high'] & distance_match['high'] & quality['high'], recommendation['medium']),

        # Medium cost preference
        ctrl.Rule(cost['medium'] & service_match['high'] & distance_match['high'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['medium'] & service_match['high'] & distance_match['high'] & quality['medium'], recommendation['high']),
        ctrl.Rule(cost['medium'] & service_match['high'] & distance_match['medium'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['medium'] & service_match['medium'] & distance_match['high'] & quality['high'], recommendation['medium']),
        ctrl.Rule(cost['medium'] & service_match['medium'] & distance_match['medium'] & quality['medium'], recommendation['medium']),
        ctrl.Rule(cost['medium'] & (service_match['low'] | distance_match['low']) & quality['high'], recommendation['low']),
        ctrl.Rule(cost['medium'] & service_match['low'] & distance_match['low'] & quality['low'], recommendation['low']),
        ctrl.Rule(cost['high'] & service_match['high'] & distance_match['high'] & quality['high'], recommendation['medium']),

        # High cost preference
        ctrl.Rule(cost['high'] & service_match['high'] & distance_match['high'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['high'] & service_match['high'] & distance_match['high'] & quality['medium'], recommendation['high']),
        ctrl.Rule(cost['high'] & service_match['high'] & distance_match['medium'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['high'] & service_match['medium'] & distance_match['high'] & quality['high'], recommendation['medium']),
        ctrl.Rule(cost['high'] & service_match['medium'] & distance_match['medium'] & quality['medium'], recommendation['medium']),
        ctrl.Rule(cost['high'] & (service_match['low'] | distance_match['low']) & quality['high'], recommendation['low']),
        ctrl.Rule(cost['high'] & service_match['low'] & distance_match['low'] & quality['low'], recommendation['low']),

        # Premium cost preference
        ctrl.Rule(cost['premium'] & service_match['high'] & distance_match['high'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['premium'] & service_match['high'] & distance_match['high'] & quality['medium'], recommendation['high']),
        ctrl.Rule(cost['premium'] & service_match['high'] & distance_match['medium'] & quality['high'], recommendation['high']),
        ctrl.Rule(cost['premium'] & service_match['medium'] & distance_match['high'] & quality['high'], recommendation['medium']),
        ctrl.Rule(cost['premium'] & service_match['medium'] & distance_match['medium'] & quality['medium'], recommendation['medium']),
        ctrl.Rule(cost['premium'] & (service_match['low'] | distance_match['low']) & quality['high'], recommendation['low']),
        ctrl.Rule(cost['premium'] & service_match['low'] & distance_match['low'] & quality['low'], recommendation['low'])
    ]

    recommender_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(recommender_ctrl)

# Main API function
def get_recommendations(location, service, cost_preference, quality_preference):
    """
    Generate hospital recommendations based on user inputs.
    
    Args:
        location (str): Preferred city or address (e.g., 'Ikorodu')
        service (str): Desired service (e.g., 'Surgery')
        cost_preference (str): Cost preference ('Low', 'Medium', 'High')
        quality_preference (str): Quality preference ('Low', 'Medium', 'High')
    
    Returns:
        dict: List of recommendations or error message
    """
    try:
        # Load and preprocess data
        df, valid_services = load_and_preprocess_data()
        
        # Validate inputs
        location = location.strip().lower() if location else 'unknown'
        service = service.strip() if service else 'unknown'
        cost_preference = get_valid_category(cost_preference, 'Medium')
        quality_preference = get_valid_category(quality_preference, 'High')
        logger.info(f'API inputs: Location={location}, Service={service}, Cost={cost_preference}, Quality={quality_preference}')
        
        # Geocode user location
        user_coords = geocode_address(location)
        if not user_coords:
            logger.warning(f'Invalid location: {location}')
            return {'error': f'Invalid location: {location.title()}'}
        
        # Filter by proximity (within 20 km to reduce load)
        df['Distance_Match'] = df['Coordinates'].apply(lambda x: compute_distance_match(user_coords, x, max_distance=20.0))
        df = df[df['Distance_Match'] > 0.0]  # Keep hospitals within 20 km
        if df.empty:
            logger.warning(f'No hospitals found within 20 km of {location}')
            return {'error': f'No hospitals found within 20 km of {location.title()}'}
        
        # Calculate service matches
        df['Service_Match'] = df['Services'].apply(lambda x: compute_service_match(service, x, valid_services))
        logger.info('Service and distance matches computed')
        
        # Debug: Log intermediate values
        logger.info('\nIntermediate Values:\n' + df[['Name', 'Services', 'Cost_Numeric', 'Quality_Numeric', 'Service_Match', 'Distance_Match']].to_string(index=False))
        
        # Setup and run fuzzy system
        recommender = setup_fuzzy_system(df)
        df['Recommendation_Score'] = df.apply(lambda x: get_recommendation_score(x, recommender), axis=1)
        
        # Filter and sort results
        recommended_hospitals = df[
            ['Name', 'Full Address', 'City', 'Services', 'Cost Level', 'Quality Score', 'User Rating', 'Recommendation_Score']
        ].sort_values(by='Recommendation_Score', ascending=False).head(10)
        
        # Save to CSV for debugging
        recommended_hospitals.to_csv('recommended_hospitals.csv', index=False)
        logger.info('Recommendations saved to recommended_hospitals.csv')
        
        # Return as JSON-serializable dict
        return recommended_hospitals.to_dict(orient='records')
    
    except Exception as e:
        logger.error(f'Error in get_recommendations: {str(e)}')
        return {'error': str(e)}

def get_recommendation_score(hospital, recommender):
    try:
        recommender.input['cost'] = hospital['Cost_Numeric']
        recommender.input['quality'] = hospital['Quality_Numeric']
        recommender.input['service_match'] = hospital['Service_Match']
        recommender.input['distance_match'] = hospital['Distance_Match']  # Updated
        recommender.compute()
        score = recommender.output['recommendation']
        logger.info(f'Hospital: {hospital["Name"]}, Service Match: {hospital["Service_Match"]:.2f}, Distance Match: {hospital["Distance_Match"]:.2f}, Score: {score:.3f}')
        return score
    except Exception as e:
        logger.error(f'Error processing {hospital["Name"]}: {str(e)}')
        return 0.0
        