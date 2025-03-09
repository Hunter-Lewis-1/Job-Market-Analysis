#!/usr/bin/env python3
"""
Traba Job Market Analysis Tool - Main script
Analyzes gig economy job markets to identify wage and demand trends
"""
import os
import time
import asyncio
from datetime import datetime
import pandas as pd
from pathlib import Path

from src.data_collection import collect_jobs_async
from src.data_processing import process_job_data, identify_key_skills
from src.analysis import analyze_market_data
from src.visualization import generate_visualizations, create_report

# Configuration
CONFIG = {
    "cities": [
        "Chicago", "Miami", "Los Angeles", "Dallas", "Atlanta", "New York"
    ],
    "sectors": [
        "warehouse", "event staff", "hospitality", 
        "distribution center", "production"
    ],
    "jobs_per_search": 20,
    "data_dir": "data",
    "output_dir": "output"
}

def setup_directories():
    """Create necessary project directories if they don't exist."""
    for dir_path in [
        CONFIG["data_dir"], 
        f"{CONFIG['data_dir']}/raw", 
        f"{CONFIG['data_dir']}/processed",
        CONFIG["output_dir"], 
        f"{CONFIG['output_dir']}/charts"
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

async def run_analysis():
    """Run the complete job market analysis workflow."""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    print(f"Job Market Analysis - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    setup_directories()
    
    # Step 1: Collect job data asynchronously
    print(f"[1/4] Collecting job data for {len(CONFIG['cities'])} cities and {len(CONFIG['sectors'])} sectors...")
    job_data = await collect_jobs_async(
        CONFIG["cities"], 
        CONFIG["sectors"],
        CONFIG["jobs_per_search"]
    )
    
    # Save raw data
    raw_data_path = f"{CONFIG['data_dir']}/raw/job_listings_{timestamp}.csv"
    job_df = pd.DataFrame(job_data)
    job_df.to_csv(raw_data_path, index=False)
    print(f"      ✓ Collected {len(job_df)} job listings")
    
    # Step 2: Process data
    print("[2/4] Processing and enriching job data...")
    processed_df = process_job_data(job_df)
    skill_data = identify_key_skills(processed_df)
    processed_path = f"{CONFIG['data_dir']}/processed/processed_jobs_{timestamp}.csv"
    processed_df.to_csv(processed_path, index=False)
    
    # Step 3: Analyze data
    print("[3/4] Analyzing market data...")
    analysis_results = analyze_market_data(processed_df)
    
    # Step 4: Create visualizations and report
    print("[4/4] Generating visualizations and final report...")
    charts = generate_visualizations(
        analysis_results,
        f"{CONFIG['output_dir']}/charts",
        timestamp
    )
    
    report_path = create_report(
        analysis_results, 
        charts,
        skill_data,
        f"{CONFIG['output_dir']}/job_market_analysis_{timestamp}.pdf"
    )
    
    execution_time = time.time() - start_time
    print(f"\nAnalysis complete in {execution_time:.2f} seconds")
    print(f"Report saved to: {report_path}")
    print(f"Data saved to: {processed_path}")
    return 0

def main():
    """Entry point for the job market analysis tool."""
    return asyncio.run(run_analysis())

if __name__ == "__main__":
    main()
