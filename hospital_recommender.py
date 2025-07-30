import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import logging
import re
import os

# Setup logging
logging.basicConfig(filename='hospital_recommender.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

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

# Load and preprocess dataset
def load_and_preprocess_data():
    dataset_path = "Lagos_hospital.csv"  # Updated to root directory
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    df = pd.read_csv(dataset_path)
    df['Cost_Numeric'] = df['Cost Level'].apply(map_cost_rating)
    df['Quality_Numeric'] = pd.to_numeric(df['Quality Score'], errors='coerce').fillna(3.0)
    df['Services'] = df['Services'].fillna('Unknown')
    df['Full Address'] = df['Full Address'].fillna('Unknown')
    
    # Dynamic city extraction
    cities = (
        r'Ikorodu|Ikoyi|Ikeja|Victoria Island|Surulere|Badagry|Lagos Island|Agege|'
        r'Alimosho|Apapa|Epe|Eti-Osa|Ibeju-Lekki|Ifako-Ijaiye|Kosofe|Lagos Mainland|'
        r'Mushin|Ojo|Oshodi-Isolo|Shomolu|Ajeromi-Ifelodun|Amuwo-Odofin|'
        r'Lekki|Ajah|Yaba|Gbagada|Maryland|Ilupeju|Ketu|Magodo|Ojota|Egbeda|'
        r'Idimu|Ipaja|Bariga|Festac Town|Amuwo|Isolo|Okota|Ikotun'
    )
    df['City'] = df['Full Address'].str.extract(f'({cities})', flags=re.IGNORECASE, expand=False)
    df['City'] = df['City'].fillna(df['Full Address'].str.split(',').str[-1].str.strip())
    df['City'] = df['City'].str.lower().replace('', 'unknown').fillna('unknown')
    logger.info('Dataset loaded and preprocessed')
    
    # Log unique cities
    unique_cities = df['City'].unique()
    logger.info(f'Extracted cities: {unique_cities}')
    
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
    location_match = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'location_match')
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

    location_match['low'] = fuzz.trimf(location_match.universe, [0, 0, 0.5])
    location_match['medium'] = fuzz.trimf(location_match.universe, [0.3, 0.5, 0.7])
    location_match['high'] = fuzz.trimf(location_match.universe, [0.5, 1, 1])

    recommendation['low'] = fuzz.trimf(recommendation.universe, [0, 0, 0.4])
    recommendation['medium'] = fuzz.trimf(recommendation.universe, [0.3, 0.5, 0.6])
    recommendation['high'] = fuzz.trimf(recommendation.universe, [0.7, 0.85, 1])

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

# Main API function
def get_recommendations(location, service, cost_preference, quality_preference):
    """
    Generate hospital recommendations based on user inputs.
    
    Args:
        location (str): Preferred city (e.g., 'Ikorodu')
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
        
        # Filter by location
        df = df[df['City'].str.lower() == location]
        if df.empty:
            logger.warning(f'No hospitals found in {location}')
            return {'error': f'No hospitals found in {location.title()}'}
        
        # Calculate matches
        df['Service_Match'] = df['Services'].apply(lambda x: compute_service_match(service, x, valid_services))
        df['Location_Match'] = 1.0
        logger.info('Service and location matches computed')
        
        # Debug: Log intermediate values
        logger.info('\nIntermediate Values:\n' + df[['Name', 'Services', 'Cost_Numeric', 'Quality_Numeric', 'Service_Match']].to_string(index=False))
        
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
        recommender.input['location_match'] = hospital['Location_Match']
        recommender.compute()
        score = recommender.output['recommendation']
        logger.info(f'Hospital: {hospital["Name"]}, Service Match: {hospital["Service_Match"]:.2f}, Score: {score:.3f}')
        return score
    except Exception as e:
        logger.error(f'Error processing {hospital["Name"]}: {str(e)}')
        return 0.0
