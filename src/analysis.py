"""
Analysis module for job market data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import stats

def analyze_wage_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze wage trends across cities and sectors.
    
    Args:
        df: Processed DataFrame with wage data
        
    Returns:
        Dictionary with wage analysis results
    """
    # Filter to entries with valid wage data
    wage_data = df[df['avg_wage'].notna()]
    
    if wage_data.empty:
        return {
            'overall': {'mean': 0, 'median': 0, 'count': 0},
            'by_city': [],
            'by_sector': [],
            'outliers': []
        }
    
    # Overall wage statistics
    overall_stats = {
        'mean': wage_data['avg_wage'].mean(),
        'median': wage_data['avg_wage'].median(),
        'std': wage_data['avg_wage'].std(),
        'min': wage_data['avg_wage'].min(),
        'max': wage_data['avg_wage'].max(),
        'count': len(wage_data),
        'pct_with_wage': len(wage_data) / len(df) * 100
    }
    
    # Calculate wage statistics by city using groupby
    # Much more efficient than manual aggregation
    city_stats = wage_data.groupby('normalized_city').agg({
        'avg_wage': ['mean', 'median', 'std', 'min', 'max', 'count']
    }).reset_index()
    
    city_stats.columns = ['city', 'mean', 'median', 'std', 'min', 'max', 'count']
    city_stats = city_stats.sort_values('mean', ascending=False)
    
    # Calculate wage statistics by sector
    sector_stats = wage_data.groupby('sector').agg({
        'avg_wage': ['mean', 'median', 'std', 'min', 'max', 'count']
    }).reset_index()
    
    sector_stats.columns = ['sector', 'mean', 'median', 'std', 'min', 'max', 'count']
    sector_stats = sector_stats.sort_values('mean', ascending=False)
    
    # Find wage outliers (using Z-score)
    z_scores = stats.zscore(wage_data['avg_wage'])
    outliers = wage_data[abs(z_scores) > 2]
    
    # City vs. sector matrix
    pivot_data = wage_data.pivot_table(
        values='avg_wage',
        index='normalized_city',
        columns='sector',
        aggfunc='mean'
    ).reset_index()
    
    return {
        'overall': overall_stats,
        'by_city': city_stats.to_dict('records'),
        'by_sector': sector_stats.to_dict('records'),
        'outliers': outliers[['title', 'company', 'normalized_city', 'sector', 'avg_wage']].to_dict('records'),
        'matrix': {
            'data': pivot_data.to_dict('list'),
            'index': pivot_data['normalized_city'].tolist(),
            'columns': [col for col in pivot_data.columns if col != 'normalized_city']
        }
    }

def analyze_demand_gaps(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze job demand patterns based on posting age and volume.
    
    Args:
        df: Processed DataFrame with job data
        
    Returns:
        Dictionary with demand gap analysis
    """
    # Fill missing days_ago with sensible default
    df_with_days = df.copy()
    df_with_days['days_ago'] = df_with_days['days_ago'].fillna(0)
    
    # Overall demand statistics
    overall_stats = {
        'total_jobs': len(df_with_days),
        'avg_days_posted': df_with_days['days_ago'].mean(),
        'recent_postings': sum(df_with_days['days_ago'] <= 3),
        'old_postings': sum(df_with_days['days_ago'] > 14)
    }
    
    # Demand by city
    city_demand = df_with_days.groupby('normalized_city').agg({
        'job_id': 'count',
        'days_ago': ['mean', 'median', 'max']
    }).reset_index()
    
    city_demand.columns = ['city', 'job_count', 'avg_days_posted', 'median_days_posted', 'max_days_posted']
    city_demand['fill_speed_rank'] = city_demand['avg_days_posted'].rank()
    city_demand['volume_rank'] = city_demand['job_count'].rank(ascending=False)
    city_demand['opportunity_score'] = (city_demand['fill_speed_rank'] + city_demand['volume_rank'])
    city_demand = city_demand.sort_values('opportunity_score')
    
    # Demand by sector
    sector_demand = df_with_days.groupby('sector').agg({
        'job_id': 'count',
        'days_ago': ['mean', 'median', 'max']
    }).reset_index()
    
    sector_demand.columns = ['sector', 'job_count', 'avg_days_posted', 'median_days_posted', 'max_days_posted']
    sector_demand['fill_speed_rank'] = sector_demand['avg_days_posted'].rank()
    sector_demand['volume_rank'] = sector_demand['job_count'].rank(ascending=False)
    sector_demand['opportunity_score'] = (sector_demand['fill_speed_rank'] + sector_demand['volume_rank'])
    sector_demand = sector_demand.sort_values('opportunity_score')
    
    # Job type analysis
    job_type_analysis = df_with_days.groupby(['job_type']).agg({
        'job_id': 'count',
        'days_ago': 'mean',
        'avg_wage': lambda x: x.mean() if x.notna().any() else None
    }).reset_index()
    
    job_type_analysis.columns = ['job_type', 'count', 'avg_days_posted', 'avg_wage']
    job_type_analysis = job_type_analysis.sort_values('count', ascending=False)
    
    # City-sector combination analysis
    city_sector = df_with_days.groupby(['normalized_city', 'sector']).agg({
        'job_id': 'count',
        'days_ago': 'mean'
    }).reset_index()
    
    city_sector.columns = ['city', 'sector', 'job_count', 'avg_days_posted']
    
    # Find top opportunities (high count, high days_ago = unfilled demand)
    city_sector['opportunity_score'] = city_sector['job_count'] * city_sector['avg_days_posted']
    top_opportunities = city_sector.sort_values('opportunity_score', ascending=False).head(10)
    
    return {
        'overall': overall_stats,
        'by_city': city_demand.to_dict('records'),
        'by_sector': sector_demand.to_dict('records'),
        'by_job_type': job_type_analysis.to_dict('records'),
        'top_opportunities': top_opportunities.to_dict('records')
    }

def analyze_skill_needs(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze skill requirements and their correlation with wages.
    
    Args:
        df: Processed DataFrame with skills data
        
    Returns:
        Dictionary with skill analysis results
    """
    # Create exploded view for skill-level analysis
    skills_df = df.explode('skills').dropna(subset=['skills'])
    
    if skills_df.empty:
        return {
            'overall': {'total_unique_skills': 0},
            'top_skills': [],
            'skills_by_sector': [],
            'skills_with_wages': []
        }
    
    # Overall skill statistics
    all_skills = df['skills'].explode().dropna()
    unique_skills = all_skills.unique()
    
    overall_stats = {
        'total_unique_skills': len(unique_skills),
        'avg_skills_per_job': df['skills'].apply(len).mean(),
        'jobs_with_skills': len(df[df['skills'].apply(len) > 0]),
        'pct_jobs_with_skills': len(df[df['skills'].apply(len) > 0]) / len(df) * 100
    }
    
    # Top skills overall
    skill_counts = all_skills.value_counts().head(20).reset_index()
    skill_counts.columns = ['skill', 'count']
    
    # Skills by sector
    sector_skills = {}
    for sector, group in df.groupby('sector'):
        sector_all_skills = group['skills'].explode().dropna()
        if len(sector_all_skills) > 0:
            top_sector_skills = sector_all_skills.value_counts().head(10).reset_index()
            top_sector_skills.columns = ['skill', 'count']
            sector_skills[sector] = top_sector_skills.to_dict('records')
    
    # Skills with highest average wages
    wage_by_skill = skills_df[skills_df['avg_wage'].notna()].groupby('skills').agg({
        'avg_wage': ['mean', 'count']
    }).reset_index()
    
    wage_by_skill.columns = ['skill', 'avg_wage', 'job_count']
    wage_by_skill = wage_by_skill[wage_by_skill['job_count'] >= 3]  # Require at least 3 jobs for significance
    high_wage_skills = wage_by_skill.sort_values('avg_wage', ascending=False).head(15)
    
    return {
        'overall': overall_stats,
        'top_skills': skill_counts.to_dict('records'),
        'skills_by_sector': sector_skills,
        'skills_with_wages': high_wage_skills.to_dict('records')
    }

def analyze_market_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform complete market analysis combining wages, demand, and skills.
    
    Args:
        df: Processed DataFrame with job data
        
    Returns:
        Dictionary with complete analysis results
    """
    analysis_results = {
        'wage_analysis': analyze_wage_trends(df),
        'demand_analysis': analyze_demand_gaps(df),
        'skill_analysis': analyze_skill_needs(df),
        'job_count': len(df),
        'city_count': df['normalized_city'].nunique(),
        'sector_count': df['sector'].nunique(),
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }
    
    # Calculate top markets (city/sector combinations with highest opportunity)
    if not df.empty:
        # Combine wage and demand data
        market_data = df.groupby(['normalized_city', 'sector']).agg({
            'job_id': 'count',
            'days_ago': 'mean',
            'avg_wage': lambda x: x.mean() if x.notna().any() else None
        }).reset_index()
        
        market_data.columns = ['city', 'sector', 'job_count', 'avg_days_posted', 'avg_wage']
        
        # Calculate opportunity score
        # Normalize factors to 0-1 scale
        if len(market_data) > 1:
            market_data['norm_count'] = (market_data['job_count'] - market_data['job_count'].min()) / \
                               (market_data['job_count'].max() - market_data['job_count'].min() + 1e-10)
            market_data['norm_days'] = (market_data['avg_days_posted'] - market_data['avg_days_posted'].min()) / \
                              (market_data['avg_days_posted'].max() - market_data['avg_days_posted'].min() + 1e-10)
            market_data['norm_wage'] = (market_data['avg_wage'] - market_data['avg_wage'].min()) / \
                              (market_data['avg_wage'].max() - market_data['avg_wage'].min() + 1e-10)
            
            # Replace NaN with 0
            market_data.fillna(0, inplace=True)
            
            # Calculate opportunity score (higher is better)
            market_data['opportunity_score'] = (market_data['norm_count'] * 0.3 + 
                                               market_data['norm_days'] * 0.3 + 
                                               market_data['norm_wage'] * 0.4)
            
            top_markets = market_data.sort_values('opportunity_score', ascending=False).head(10)
            analysis_results['top_markets'] = top_markets.to_dict('records')
    
    return analysis_results
