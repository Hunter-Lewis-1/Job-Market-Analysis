from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    """Analyzes the sentiment of a text using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score > 0.05:
        sentiment = 'positive'
    elif compound_score < -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return sentiment, compound_score, scores
