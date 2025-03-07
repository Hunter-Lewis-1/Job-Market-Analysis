import pandas as pd

def aggregate_data(news_data):
    """Creates DataFrame from news data."""
    if not news_data:
        return pd.DataFrame()
    news_df = pd.DataFrame(news_data)
    news_df['source'] = 'News'
    if 'publication_date' in news_df.columns:
        news_df['publication_date'] = pd.to_datetime(news_df['publication_date'], errors='coerce')
    news_df['date'] = news_df['publication_date']
    return news_df

def calculate_sentiment_metrics(df, company_name):
    """Calculates average sentiment and percentage breakdown."""
    if df.empty:
        return 0, 0, 0, 0
    avg_sentiment = df['compound_score'].mean()
    positive_percentage = (df['sentiment'] == 'positive').sum() / len(df) * 100
    negative_percentage = (df['sentiment'] == 'negative').sum() / len(df) * 100
    neutral_percentage = (df['sentiment'] == 'neutral').sum() / len(df) * 100
    return avg_sentiment, positive_percentage, negative_percentage, neutral_percentage

def calculate_sentiment_by_year(df, company_name):
    """Groups articles by year and calculates sentiment metrics for each year."""
    if df.empty:
        return []
    
    # Ensure date column is datetime
    df['year'] = pd.to_datetime(df['publication_date'], errors='coerce').dt.year
    
    # Group by year and calculate metrics
    results = []
    for year, year_df in df.groupby('year'):
        if year and not pd.isna(year):  # Skip NaN years
            avg_sentiment, pos_perc, neg_perc, neu_perc = calculate_sentiment_metrics(year_df, company_name)
            results.append({
                'company': company_name,
                'year': int(year),
                'average_sentiment': avg_sentiment,
                'positive_percentage': pos_perc,
                'negative_percentage': neg_perc,
                'neutral_percentage': neu_perc,
                'article_count': len(year_df)
            })
    
    return results
