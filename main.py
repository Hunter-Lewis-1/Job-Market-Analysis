import os
import pandas as pd
from src.data_collection import collect_news_articles
from src.data_preprocessing import clean_text
from src.sentiment_analysis import analyze_sentiment
from src.topic_modeling import extract_topics
from src.analysis import aggregate_data, calculate_sentiment_metrics, calculate_sentiment_by_year
from src.visualization import plot_sentiment_comparison, generate_report

def main():
    """Main function with debug checks and dummy data option."""
    for dir_path in ["data/raw", "data/processed", "data/results", "reports", "visualizations"]:
        os.makedirs(dir_path, exist_ok=True)

    companies = ['Traba', 'Instawork', 'Wonolo', 'Ubeya', 'Sidekicker', 'Zenjob', 'Veryable', 'Shiftgig']
    all_sentiment_metrics = []
    all_topics = {}
    use_dummy_data = False  # Set to True to test with dummy data

    # Step 1: Data Collection
    all_news_data = {}
    for company in companies:
        print(f"Collecting maximum news articles from Google News for {company}...")
        news_data = collect_news_articles(company, use_dummy=use_dummy_data)
        all_news_data[company] = news_data
        print(f"{company}: Collected {len(news_data)} articles")
    print("Step 1: Data Collection Complete. Sample data for Traba:")
    traba_sample = pd.DataFrame(all_news_data['Traba']).head().to_string() if all_news_data['Traba'] else "No data"
    print(traba_sample)

    # Step 2: Preprocessing and Sentiment Analysis
    all_processed_data = {}
    all_tokenized_texts = {}
    for company, news_data in all_news_data.items():
        print(f"Processing {company}...")
        if not news_data:
            print(f"No news data for {company}")
            all_processed_data[company] = pd.DataFrame()
            all_tokenized_texts[company] = []
            continue
        for item in news_data:
            if 'text' in item:
                item['cleaned_text'] = clean_text(item['text'])
                item['sentiment'], item['compound_score'], item['keyword_scores'] = analyze_sentiment(item['cleaned_text'])
            else:
                print(f"Warning: No 'text' key in article for {company}")
        all_processed_data[company] = aggregate_data(news_data)
        if all_processed_data[company].empty:
            all_tokenized_texts[company] = []
        else:
            all_tokenized_texts[company] = all_processed_data[company]['cleaned_text'].dropna().tolist()
            
        # Calculate sentiment by year
        yearly_metrics = calculate_sentiment_by_year(all_processed_data[company], company)
        all_sentiment_metrics.extend(yearly_metrics)
            
    print("Step 2: Preprocessing and Sentiment Analysis Complete. Sample for Traba:")
    traba_processed = all_processed_data['Traba'].head().to_string() if not all_processed_data['Traba'].empty else "No data"
    print(traba_processed)

    # Step 3: Topic Modeling
    for company in companies:
        if all_tokenized_texts[company]:
            all_topics[company] = extract_topics(all_tokenized_texts[company], num_topics=5, top_n=10)
        else:
            all_topics[company] = []
            print(f"No topics for {company} due to empty tokenized texts")
    print("Step 3: Topic Modeling Complete. Sample topics for Traba:")
    if 'Traba' in all_topics and all_topics['Traba']:
        for topic_name, words in all_topics['Traba'][:3]:
            print(f"{topic_name}: {', '.join(words)}")
    else:
        print("No topics for Traba")

    # Step 4: Analysis and Data Export
    for company in companies:
        df = all_processed_data[company]
        if not df.empty:
            # Add year column
            df['year'] = pd.to_datetime(df['publication_date'], errors='coerce').dt.year
            # Export processed data with year
            df.to_csv(f"data/processed/{company.lower()}_processed_data.csv", index=False)
            # Export sentiment metrics by year
            metrics_df = pd.DataFrame([m for m in all_sentiment_metrics if m['company'] == company])
            metrics_df.to_csv(f"data/results/{company.lower()}_sentiment_by_year.csv", index=False)
        else:
            print(f"No data to export for {company}")
            
    print("Step 4: Analysis Complete. Sentiment metrics by year:")
    metrics_summary = pd.DataFrame(all_sentiment_metrics)
    print(metrics_summary.to_string() if not metrics_summary.empty else "Empty DataFrame")

    # Step 5: Visualization and Reporting
    plot_sentiment_comparison(all_sentiment_metrics, "visualizations/sentiment_comparison.html")
    generate_report(all_sentiment_metrics, all_topics, "reports/sentiment_analysis_report.pdf")
    print("Step 5: Visualization and Reporting Complete. Files saved:")
    print("- visualizations/sentiment_comparison.html")
    print("- visualizations/sentiment_comparison_distribution.html")
    print("- reports/sentiment_analysis_report.pdf")
    print("Project execution completed.")

if __name__ == "__main__":
    main()
