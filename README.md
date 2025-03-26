# Job Market Analysis Tool

This project analyzes gig economy job markets by scraping job board data to identify wage trends, skill demands, and market opportunities - providing strategic insights for Gig Economy business operations.

## Features

- Collects job listings from Indeed for multiple cities and gig economy sectors
- Uses asynchronous scraping for optimal performance
- Normalizes wage data for consistent analysis
- Extracts in-demand skills from job descriptions
- Analyzes market opportunities based on wages, job volume, and fill rates
- Generates interactive visualizations and comprehensive PDF reports

## To Run

1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python main.py`

## Files

- `requirements.txt`: Lists required packages
- `src/`: Contains the Python code
  - `data_collection.py`: Scrapes job listings from job boards
  - `data_processing.py`: Cleans and normalizes job data
  - `analysis.py`: Analyzes wage trends, demand gaps, and skills
  - `visualization.py`: Creates visualizations and reports
- `data/`: Stores data (raw and processed)
- `output/`: Contains charts and the final report

## Outputs

- Wage heatmap by city and sector (HTML interactive)
- Job demand by city and sector charts
- Skill frequency and wage correlation visualizations
- Market opportunity bubble chart
- Comprehensive PDF report with analysis and recommendations

## Performance Optimizations

- Asynchronous scraping with rate limiting for efficient data collection
- Vectorized pandas operations for fast data processing
- Memory-efficient data structures for large job datasets
- Smart caching of intermediate results
