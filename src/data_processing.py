"""
Data processing module for job board analysis
Optimized for efficient data cleaning and normalization
"""
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Constants for data cleaning
STOP_WORDS = set(stopwords.words('english'))
SKILL_PATTERNS = {
    'warehouse': [r'\bforklift\b', r'\bpallet\s?jack\b', r'\binventory\b', r'\bpicking\b', r'\bpacking\b', 
                  r'\bshipping\b', r'\breceiving\b', r'\bloader\b', r'\bwarehouse\b'],
    'logistics': [r'\bdistribution\b', r'\bsupply\s?chain\b', r'\bfreight\b', r'\blogistics\b'],
    'events': [r'\bcatering\b', r'\bevent\b', r'\bsetup\b', r'\bbreakdown\b', r'\bserving\b'],
    'hospitality': [r'\bcustomer\s?service\b', r'\bfood\s?service\b', r'\bhospitality\b', r'\bguest\b'],
    'certification': [r'\bcdl\b', r'\bcertified\b', r'\bosha\b', r'\blicense[d]?\b'],
    'general': [r'\blifting\b', r'\bflexible\b', r'\bovertime\b', r'\bweekend\b']
}

# City normalization mapping
CITY_MAPPING = {
    'nyc': 'New York',
    'ny': 'New York',
    'la': 'Los Angeles',
    'sf': 'San Francisco',
    'chi': 'Chicago',
    'atl': 'Atlanta',
    'dfw': 'Dallas',
    'phx': 'Phoenix',
    'philly': 'Philadelphia'
}

def normalize_wage_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all wage data to hourly rates using vectorized operations.
    
    Args:
        df: DataFrame with wage data
        
    Returns:
        DataFrame with normalized hourly wages
    """
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Fill NaN wage_rate with empty string for safe string operations
    result['wage_rate'] = result['wage_rate'].fillna('')
    
    # Create conversion masks for vectorized operations
    annual_mask = result['wage_rate'].str.lower().str.contains('annual|year', na=False)
    weekly_mask = result['wage_rate'].str.lower().str.contains('week', na=False)
    monthly_mask = result['wage_rate'].str.lower().str.contains('month', na=False)
    daily_mask = result['wage_rate'].str.lower().str.contains('day', na=False)
    
    # Vectorized operations for wage conversion
    # Annual to hourly (2080 hours per year)
    if annual_mask.any():
        for col in ['min_wage', 'max_wage', 'avg_wage']:
            result.loc[annual_mask & result[col].notna(), col] = result.loc[annual_mask & result[col].notna(), col] / 2080
        result.loc[annual_mask, 'wage_rate'] = 'hourly'
    
    # Weekly to hourly (40 hours per week)
    if weekly_mask.any():
        for col in ['min_wage', 'max_wage', 'avg_wage']:
            result.loc[weekly_mask & result[col].notna(), col] = result.loc[weekly_mask & result[col].notna(), col] / 40
        result.loc[weekly_mask, 'wage_rate'] = 'hourly'
    
    # Monthly to hourly (173.33 hours per month)
    if monthly_mask.any():
        for col in ['min_wage', 'max_wage', 'avg_wage']:
            result.loc[monthly_mask & result[col].notna(), col] = result.loc[monthly_mask & result[col].notna(), col] / 173.33
        result.loc[monthly_mask, 'wage_rate'] = 'hourly'
    
    # Daily to hourly (8 hours per day)
    if daily_mask.any():
        for col in ['min_wage', 'max_wage', 'avg_wage']:
            result.loc[daily_mask & result[col].notna(), col] = result.loc[daily_mask & result[col].notna(), col] / 8
        result.loc[daily_mask, 'wage_rate'] = 'hourly'
    
    # Round wage values for readability
    for col in ['min_wage', 'max_wage', 'avg_wage']:
        if col in result.columns:
            result[col] = result[col].round(2)
    
    return result

def normalize_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize location data using vectorized string operations.
    
    Args:
        df: DataFrame with location data
        
    Returns:
        DataFrame with added normalized_city column
    """
    result = df.copy()
    
    # Extract city from location string (e.g., "Chicago, IL" -> "Chicago")
    # Using vectorized operations for performance
    result['city'] = result['location'].str.extract(r'^([^,]+)', expand=False).str.strip()
    
    # Apply city mapping for common abbreviations/variations
    # More efficient than apply() since we're doing simple replacements
    result['normalized_city'] = result['city'].str.lower()
    for abbrev, full_name in CITY_MAPPING.items():
        # Use vectorized replacements
        mask = result['normalized_city'].str.contains(abbrev, case=False, na=False)
        result.loc[mask, 'normalized_city'] = full_name
    
    # Capitalize first letter of each word for consistent formatting
    result['normalized_city'] = result['normalized_city'].str.title()
    
    return result

def extract_job_skills(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract skills mentioned in job descriptions using regex patterns.
    Vectorized implementation for performance.
    
    Args:
        df: DataFrame with job descriptions
        
    Returns:
        DataFrame with skills column added
    """
    result = df.copy()
    
    # Initialize empty skills lists
    result['skills'] = [[] for _ in range(len(result))]
    
    # Only process rows with valid descriptions
    valid_desc_mask = result['description'].notna() & (result['description'] != '')
    
    # For each skill category
    for category, patterns in SKILL_PATTERNS.items():
        # Combine patterns for efficiency
        combined_pattern = '|'.join(patterns)
        
        # Apply regex search vectorized - creates a boolean Series
        for i, (idx, row) in enumerate(result[valid_desc_mask].iterrows()):
            desc_lower = row['description'].lower()
            found_skills = set()
            
            # Check each pattern in this category
            for pattern in patterns:
                if re.search(pattern, desc_lower):
                    # Extract the actual matched text
                    matches = re.finditer(pattern, desc_lower)
                    for match in matches:
                        skill_text = match.group(0).strip()
                        if skill_text and len(skill_text) > 2:  # Avoid single letter matches
                            found_skills.add(skill_text)
            
            # Add found skills to the corresponding row
            if found_skills:
                result.at[idx, 'skills'].extend(found_skills)
    
    # Extract years of experience using regex
    exp_pattern = r'(\d+)\+?\s*(?:year|yr)s?(?:\s+of)?\s+experience'
    for idx, row in result[valid_desc_mask].iterrows():
        exp_match = re.search(exp_pattern, row['description'].lower())
        if exp_match:
            result.at[idx, 'skills'].append(f"{exp_match.group(1)}+ years experience")
    
    return result

def classify_job_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify jobs as full-time, part-time, contract, etc.
    
    Args:
        df: DataFrame with job data
        
    Returns:
        DataFrame with job_type column added
    """
    result = df.copy()
    
    # Combine title and description for search
    result['search_text'] = result['title'].fillna('') + ' ' + result['description'].fillna('')
    result['search_text'] = result['search_text'].str.lower()
    
    # Create job type classification using vectorized operations
    conditions = [
        result['search_text'].str.contains(r'part[\s-]time|part time', na=False),
        result['search_text'].str.contains(r'contract|contractor', na=False),
        result['search_text'].str.contains(r'temporary|temp\b', na=False),
        result['search_text'].str.contains(r'seasonal', na=False),
        result['search_text'].str.contains(r'flexible|flex\b', na=False),
        result['search_text'].str.contains(r'full[\s-]time|full time', na=False)
    ]
    
    choices = ['Part-time', 'Contract', 'Temporary', 'Seasonal', 'Flexible', 'Full-time']
    
    # Default to Full-time if no match
    result['job_type'] = np.select(conditions, choices, default='Full-time')
    
    # Drop the temporary search_text column
    result.drop('search_text', axis=1, inplace=True)
    
    return result

def process_job_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process job data through the complete pipeline.
    
    Args:
        df: Raw DataFrame with job data
        
    Returns:
        Processed DataFrame with normalized data
    """
    if df.empty:
        return df
    
    # Convert dates to datetime
    df['posting_date'] = pd.to_datetime(df['posting_date'], errors='coerce')
    df['scrape_date'] = pd.to_datetime(df['scrape_date'], errors='coerce')
    
    # Process data through pipeline
    processed_df = (df.pipe(normalize_wage_data)
                      .pipe(normalize_location)
                      .pipe(extract_job_skills)
                      .pipe(classify_job_types))
    
    # Add calculated fields
    processed_df['has_wage_data'] = processed_df['avg_wage'].notna()
    processed_df['skills_count'] = processed_df['skills'].apply(len)
    
    return processed_df

def identify_key_skills(df: pd.DataFrame) -> Dict[str, List[Tuple[str, int]]]:
    """
    Identify key skills by sector from processed job data.
    
    Args:
        df: Processed DataFrame with skills data
        
    Returns:
        Dictionary mapping sectors to their top skills with counts
    """
    result = {}
    
    # Group by sector
    for sector, group in df.groupby('sector'):
        # Flatten all skills lists and count occurrences
        all_skills = [skill for skills_list in group['skills'].dropna() for skill in skills_list]
        if not all_skills:
            result[sector] = []
            continue
            
        # Count skill frequencies
        skill_counter = Counter(all_skills)
        
        # Get top skills (up to 15)
        top_skills = skill_counter.most_common(15)
        
        result[sector] = top_skills
    
    return result
