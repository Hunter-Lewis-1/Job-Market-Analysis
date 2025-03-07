import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def plot_sentiment_comparison(sentiment_metrics_list, filepath):
    """
    Creates a line chart showing sentiment trends over time for each company.
    
    Args:
        sentiment_metrics_list: List of dicts with company, year, and sentiment metrics
        filepath: Where to save the HTML visualization
    """
    if not sentiment_metrics_list:
        print("No data available for sentiment visualization.")
        return
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame(sentiment_metrics_list)
    
    # Create line chart of sentiment over time
    fig = px.line(
        df,
        x='year',
        y='average_sentiment',
        color='company',
        title='Sentiment Trends by Company (2021-2025)',
        labels={'average_sentiment': 'Average Sentiment Score', 'year': 'Year'},
        markers=True
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Average Sentiment Score',
        legend_title='Company',
        hovermode='x unified'
    )
    
    # Add hover data showing article counts
    fig.update_traces(
        hovertemplate='<b>%{customdata[0]}</b><br>'
                      'Year: %{x}<br>'
                      'Sentiment: %{y:.2f}<br>'
                      'Articles: %{customdata[1]}<br>'
                      'Positive: %{customdata[2]:.1f}%<br>'
                      'Negative: %{customdata[3]:.1f}%<br>'
                      'Neutral: %{customdata[4]:.1f}%',
        customdata=df[['company', 'article_count', 'positive_percentage', 
                       'negative_percentage', 'neutral_percentage']]
    )
    
    # Save as interactive HTML
    fig.write_html(filepath)
    print(f"Sentiment trend visualization saved to {filepath}")
    
    # Create a second visualization showing sentiment distribution
    dist_file = filepath.replace('.html', '_distribution.html')
    fig2 = px.bar(
        df,
        x='company',
        y=['positive_percentage', 'neutral_percentage', 'negative_percentage'],
        color_discrete_map={
            'positive_percentage': 'green',
            'neutral_percentage': 'gray',
            'negative_percentage': 'red'
        },
        barmode='stack',
        facet_col='year',
        title='Sentiment Distribution by Company and Year'
    )
    fig2.update_layout(legend_title='Sentiment Type')
    fig2.write_html(dist_file)

def generate_report(sentiment_metrics_list, all_topics, filepath):
    """
    Generates a PDF report with sentiment metrics by year and topics.
    
    Args:
        sentiment_metrics_list: List of dicts with company, year, and sentiment metrics
        all_topics: Dict of topics by company
        filepath: Where to save the PDF report
    """
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Title
    elements.append(Paragraph("Sentiment Analysis Report (2021-2025)", title_style))
    elements.append(Spacer(1, 12))
    
    # Convert metrics to DataFrame for easier handling
    if sentiment_metrics_list:
        metrics_df = pd.DataFrame(sentiment_metrics_list)
        
        # Section: Sentiment Trends by Year
        elements.append(Paragraph("Sentiment Trends By Year", subtitle_style))
        elements.append(Spacer(1, 6))
        
        # Group by company and year
        companies = sorted(metrics_df['company'].unique())
        years = sorted(metrics_df['year'].unique())
        
        # Create a table for sentiment by year
        table_data = [["Company"] + [f"Year {year}" for year in years]]
        
        for company in companies:
            company_data = [company]
            for year in years:
                year_data = metrics_df[(metrics_df['company'] == company) & 
                                      (metrics_df['year'] == year)]
                if not year_data.empty:
                    sentiment = year_data['average_sentiment'].values[0]
                    count = year_data['article_count'].values[0]
                    company_data.append(f"{sentiment:.2f} ({count} articles)")
                else:
                    company_data.append("No data")
            table_data.append(company_data)
        
        # Create the table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))
    
    # Add topics section
    elements.append(Paragraph("Top Topics by Company", subtitle_style))
    elements.append(Spacer(1, 6))
    
    for company, topics in all_topics.items():
        if topics:
            elements.append(Paragraph(f"{company}:", styles['Heading3']))
            for i, (topic_name, words) in enumerate(topics[:3]):
                elements.append(Paragraph(f"Topic {i+1}: {', '.join(words)}", normal_style))
            elements.append(Spacer(1, 6))
        else:
            elements.append(Paragraph(f"{company}: No topics identified", normal_style))
    
    # Build the PDF
    doc.build(elements)
    print(f"Report generated at {filepath}")
