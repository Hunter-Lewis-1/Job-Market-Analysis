"""
Data processing module for cleaning and normalizing job data.
"""
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from typing import List, Dict, Any

# Download NLTK resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Define common skill keywords to extract
SKILL_KEYWORDS = {
    'warehouse': ['forklift', 'pallet jack', 'inventory', 'shipping', 'receiving', 'picking', 'packing', 
                  'loader', 'unloader', 'logistics', 'warehouse', 'scanning', 'lifting'],
    'distribution': ['sorting', 'loading', 'unloading', 'order picking', 'inventory control', 
                     'distribution', 'supply chain', 'shipping', 'receiving'],
    'event': ['customer service', 'setup', 'breakdown', 'catering', 'hospitality', 'event', 
              'serving', 'bartending', 'hosting'],
    'hospitality': ['customer service', 'food service', 'cleaning', 'housekeeping', 'front desk',
                    'guest', 'hospitality', 'hotel', 'restaurant'],
    'production': ['assembly', 'manufacturing', 'production', 'quality control', 'machine operator',
                  'packaging', 'inspection', 'fabrication']
}

# Location normalization mapping
CITY_MAPPING = {
    'nyc': 'New York',
    'ny': 'New York',
    'la': 'Los Angeles',
    'sf': 'San Francisco',
    'chi': 'Chicago',
    'atl': 'Atlanta',
    'dfw': 'Dallas',
    'phoenix az': 'Phoenix',
    'phx': 'Phoenix',
    'philly': 'Philadelphia',
    'houston tx': 'Houston',
    'miami fl': 'Miami'
}

def normalize_wage_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize wage data by converting everything to hourly rates.
    
    Args:
        df: DataFrame containing job data
        
    Returns:
        DataFrame with normalized wage data
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Convert annual salaries to hourly (assuming 2080 work hours per year)
    mask = (result['wage_rate'] == 'annual') & (result['avg_wage'].notna())
    result.loc[mask, 'avg_wage'] = result.loc[mask, 'avg_wage'] / 2080
    result.loc[mask, 'min_wage'] = result.loc[mask, 'min_wage'] / 2080
    result.loc[mask, 'max_wage'] = result.loc[mask, 'max_wage'] / 2080
    result.loc[mask, 'wage_rate'] = 'hourly'
    
    # Convert weekly salaries to hourly (assuming 40 hour work week)
    mask = (result['wage_rate'] == 'weekly') & (result['avg_wage'].notna())
    result.loc[mask, 'avg_wage'] = result.loc[mask, 'avg_wage'] / 40
    result.loc[mask, 'min_wage'] = result.loc[mask, 'min_wage'] / 40
    result.loc[mask, 'max_wage'] = result.loc[mask, 'max_wage'] / 40
    result.loc[mask, 'wage_rate'] = 'hourly'
    
    # Convert monthly salaries to hourly (assuming 173.33 work hours per month)
    mask = (result['wage_rate'] == 'monthly') & (result['avg_wage'].notna())
    result.loc[mask, 'avg_wage'] = result.loc[mask, 'avg_wage'] / 173.33
    result.loc[mask, 'min_wage'] = result.loc[mask, 'min_wage'] / 173.33
    result.loc[mask, 'max_wage'] = result.loc[mask, 'max_wage'] / 173.33
    result.loc[mask, 'wage_rate'] = 'hourly'
    
    # Convert daily salaries to hourly (assuming 8 hour work day)
    mask = (result['wage_rate'] == 'daily') & (result['avg_wage'].notna())
    result.loc[mask, 'avg_wage'] = result.loc[mask, 'avg_wage'] / 8
    result.loc[mask, 'min_wage'] = result.loc[mask, 'min_wage'] / 8
    result.loc[mask, 'max_wage'] = result.loc[mask, 'max_wage'] / 8
    result.loc[mask, 'wage_rate'] = 'hourly'
    
    return result

def normalize_location(location_str: str) -> str:
    """
    Normalize location strings to standard city names.
    
    Args:
        location_str: Location string from job posting
        
    Returns:
        Normalized city name
    """
    if pd.isna(location_str) or not location_str:
        return "Unknown"
    
    location_lower = location_str.lower()
    
    # Check for direct matches in our mapping
    for key, value in CITY_MAPPING.items():
        if key in location_lower:
            return value
            
    # Try to extract city name from format like "Chicago, IL"
    city_match = re.match(r'^([^,]+)', location_str)
    if city_match:
        return city_match.group(1).strip()
    
    return location_str

def extract_skills(description: str, sector: str) -> List[str]:
    """
    Extract relevant skills from job description.
    
    Args:
        description: Job description text
        sector: Job sector
        
    Returns:
        List of skills found in the description
    """
    if pd.isna(description) or not description:
        return []
    
    description_lower = description.lower()
    found_skills = []
    
    # Combine general skills with sector-specific skills
    skill_keywords = set()
    for key, skills in SKILL_KEYWORDS.items():
        if key in sector.lower() or sector.lower() in key:
            skill_keywords.update(skills)
    
    # Check for each skill keyword
    for skill in skill_keywords:
        if skill.lower() in description_lower:
            found_skills.append(skill)
    
    # Check for specific certifications and requirements
    certifications = ['cdl', 'forklift certification', 'osha', 'servsafe', 'food handler']
    for cert in certifications:
        if cert.lower() in description_lower:
            found_skills.append(cert)
    
    # Check for experience requirements
    experience_match = re.search(r'(\d+)\+?\s*(?:year|yr)s?(?:\s+of)?\s+experience', description_lower)
    if experience_match:
        found_skills.append(f"{experience_match.group(1)}+ years experience")
    
    # Check for education requirements
    if 'high school' in description_lower or 'ged' in description_lower:
        found_skills.append('high school/GED')
    if 'bachelor' in description_lower or 'college degree' in description_lower:
        found_skills.append('bachelor degree')
    
    return found_skills

def classify_job_type(title: str, description: str) -> str:
    """
    Classify the job as full-time, part-time, or flexible.
    
    Args:
        title: Job title
        description: Job description
        
    Returns:
        Job type classification
    """
    if pd.isna(title) and pd.isna(description):
        return "Unknown"
    
    text = f"{title} {description}".lower() if not pd.isna(description) else title.lower()
    
    if 'part-time' in text or 'part time' in text:
        return "Part-time"
    elif 'full-time' in text or 'full time' in text:
        return "Full-time"
    elif 'flexible' in text or 'flex' in text:
        return "Flexible"
    elif 'temporary' in text or 'temp' in text:
        return "Temporary"
    elif 'contract' in text:
        return "Contract"
    else:
        return "Full-time"  # Default assumption

def process_job_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process job data by cleaning and normalizing values.
    
    Args:
        df: DataFrame containing raw job data
        
    Returns:
        Processed DataFrame
    """
    # Create a copy of the DataFrame
    result = df.copy()
    
    # Convert posting_date to datetime
    result['posting_date'] = pd.to_datetime(result['posting_date'], errors='coerce')
    result['scrape_date'] = pd.to_datetime(result['scrape_date'], errors='coerce')
    
    # Normalize wage data
    result = normalize_wage_data(result)
    
    # Normalize locations
    result['normalized_location'] = result['location'].apply(normalize_location)
    
    # Extract skills from description
    result['skills'] = result.apply(lambda row: extract_skills(row['description'], row['sector']), axis=
