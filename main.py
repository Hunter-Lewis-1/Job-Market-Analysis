#!/usr/bin/env python3
"""
Traba Job Market Analysis - Main Script
Analyzes gig economy job market data to identify trends in wages and demand.
"""
import os
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from src.data_collection import collect_job_data
from src.data_processing import process_job_data
from src.analysis import (
    analyze_wage_trends, 
    analyze_demand_gaps, 
    analyze_skill_needs
)
from src.visualization import (
    generate_wage_heatmap,
    generate_demand_chart,
    generate_skill_chart,
    generate_report
)

# Configuration
CITIES = [
    "Chicago", "Miami", "Los Angeles", "Dallas", "Atlanta", 
    "New York", "Houston", "Phoenix", "Philadelphia", "San Francisco"
]
SECTORS = ["warehouse", "distribution center", "event staff", "hospitality", "production"]
OUTPUT_DIR = "output"
DATA_DIR = "data"

def setup_directories():
    """Create necessary directories if they don't exist."""
    for directory in [OUTPUT_DIR, DATA_DIR, f"{DATA_DIR}/raw", f"{DATA_DIR}/processed", f"{OUTPUT_DIR}/visualizations"]:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main execution function for the Traba Job Market Analysis."""
    start_time = datetime.now()
    print(f"Starting job market analysis: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    setup_directories()
    
    # Step 1: Collect job data (parallel for efficiency)
    print("\n1. Collecting job data from Indeed...")
    jobs = []
    
    # Using ThreadPoolExecutor for parallel scraping (improves performance)
    with ThreadPoolExecutor(max_workers=min(10, len(CITIES))) as executor:
        future_to_city = {executor.submit(collect_job_data, city, SECTORS): city for city in CITIES}
        for future in future_to_city:
            city_jobs = future.result()
            jobs.extend(city_jobs)
            print(f"  ✓ Collected {len(city_jobs)} jobs from {future_to_city[future]}")
    
    # Save raw data
    raw_df = pd.DataFrame(jobs)
    raw_df.to_csv(f"{DATA_DIR}/raw/job_listings_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
    print(f"  ✓ Total jobs collected: {len(jobs)}")
    
    # Step 2: Process the data
    print("\n2. Processing job data...")
    processed_df = process_job_data(raw_df)
    processed_df.to_csv(f"{DATA_DIR}/processed/processed_jobs_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
    print(f"  ✓ Processed {len(processed_df)} job listings")
    
    # Step 3: Analyze the data
    print("\n3. Analyzing job market data...")
    
    wage_analysis = analyze_wage_trends(processed_df)
    print("  ✓ Wage analysis complete")
    
    demand_analysis = analyze_demand_gaps(processed_df)
    print("  ✓ Demand gap analysis complete")
    
    skill_analysis = analyze_skill_needs(processed_df)
    print("  ✓ Skill needs analysis complete")
    
    # Step 4: Generate visualizations
    print("\n4. Generating visualizations...")
    
    wage_fig = generate_wage_heatmap(wage_analysis)
    wage_fig.write_html(f"{OUTPUT_DIR}/visualizations/wage_heatmap.html")
    print("  ✓ Wage heatmap generated")
    
    demand_fig = generate_demand_chart(demand_analysis)
    demand_fig.write_html(f"{OUTPUT_DIR}/visualizations/demand_chart.html")
    print("  ✓ Demand chart generated")
    
    skill_fig = generate_skill_chart(skill_analysis)
    skill_fig.write_html(f"{OUTPUT_DIR}/visualizations/skill_chart.html")
    print("  ✓ Skill chart generated")
    
    # Step 5: Generate report
    print("\n5. Generating final report...")
    report_path = generate_report(
        wage_analysis, 
        demand_analysis,
        skill_analysis,
        f"{OUTPUT_DIR}/traba_job_market_analysis.pdf"
    )
    print(f"  ✓ Report generated: {report_path}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    print(f"\nAnalysis complete! Total runtime: {duration:.2f} minutes")
    print(f"Results saved to {OUTPUT_DIR} directory")

if __name__ == "__main__":
    main()
