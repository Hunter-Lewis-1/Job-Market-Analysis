# Traba Sentiment Analysis Project 

This project analyzes the sentiment of news articles from Google News for Traba and its competitors, collecting the maximum number of articles available since 2021.

## Features

- Collects up to 500 news articles per company from Google News (since 2021)
- Uses retry logic to maximize successful text extraction
- Performs sentiment analysis using VADER
- Analyzes sentiment trends over time (by year)
- Extracts key topics using NLP
- Generates visualizations of sentiment trends
- Creates comprehensive PDF reports

## To Run

1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python main.py` 

## Files

- `requirements.txt`: Lists required packages.
- `src/`: Contains the Python code.
  - `data_collection.py`: Collects articles from Google News
  - `data_preprocessing.py`: Cleans and prepares text data
  - `sentiment_analysis.py`: Performs sentiment analysis
  - `topic_modeling.py`: Extracts topics from text
  - `analysis.py`: Calculates metrics and analyzes data
  - `visualization.py`: Creates visualizations and reports
- `data/`: Stores data (raw, processed, results).
- `reports/`: Contains the PDF report.
- `visualizations/`: Contains interactive charts.
- `main.py`: The main script.

## Outputs

- Sentiment comparison over time (HTML visualization)
- Sentiment distribution by company and year
- PDF report with sentiment metrics and key topics
- CSV files with processed data and results
