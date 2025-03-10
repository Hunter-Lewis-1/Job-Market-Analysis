"""
Visualization module for job market data
Creates charts and generates comprehensive PDF reports
"""
import os
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io

def create_wage_heatmap(wage_data: Dict[str, Any], output_dir: str, timestamp: str) -> str:
    """
    Create a heatmap of wages by city and sector.
    
    Args:
        wage_data: Wage analysis data
        output_dir: Directory to save visualization
        timestamp: Timestamp string for filename
        
    Returns:
        Path to saved visualization
    """
    # Extract matrix data
    matrix_data = wage_data['matrix']
    
    if not matrix_data['columns'] or len(matrix_data['columns']) <= 1:
        # Not enough data for heatmap
        return ""
    
    # Convert data to DataFrame for easier plotting
    z_data = []
    cities = matrix_data['index']
    sectors = [col for col in matrix_data['columns'] if col != 'normalized_city']
    
    for city in cities:
        city_idx = matrix_data['index'].index(city)
        row = []
        for sector in sectors:
            sector_idx = matrix_data['columns'].index(sector)
            value = matrix_data['data'].get(sector, [])[city_idx] if city_idx < len(matrix_data['data'].get(sector, [])) else None
            row.append(value)
        z_data.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=sectors,
        y=cities,
        colorscale='Viridis',
        colorbar=dict(title='Hourly Wage ($)'),
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Average Hourly Wages by City and Sector',
        xaxis_title='Job Sector',
        yaxis_title='City',
        height=600,
        width=900
    )
    
    # Save visualization
    output_path = os.path.join(output_dir, f'wage_heatmap_{timestamp}.html')
    fig.write_html(output_path)
    
    return output_path

def create_demand_bar_chart(demand_data: Dict[str, Any], output_dir: str, timestamp: str) -> str:
    """
    Create a bar chart of job demand by city and sector.
    
    Args:
        demand_data: Demand analysis data
        output_dir: Directory to save visualization
        timestamp: Timestamp string for filename
        
    Returns:
        Path to saved visualization
    """
    # Extract city demand data
    city_data = pd.DataFrame(demand_data['by_city'])
    if city_data.empty:
        return ""
    
    # Sort by job count
    city_data = city_data.sort_values('job_count', ascending=False)
    
    # Create bar chart with custom hover text
    fig = px.bar(
        city_data.head(10),
        x='city',
        y='job_count',
        color='avg_days_posted',
        color_continuous_scale='Viridis',
        labels={'city': 'City', 'job_count': 'Number of Jobs', 'avg_days_posted': 'Avg Days Posted'},
        title='Job Demand by City (Top 10)'
    )
    
    fig.update_layout(
        xaxis_title='City',
        yaxis_title='Number of Job Listings',
        height=500,
        width=900,
        coloraxis_colorbar=dict(title='Avg Days Posted'),
    )
    
    # Add text annotations for days posted
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Jobs: %{y}<br>Avg Days Posted: %{marker.color:.1f}<extra></extra>'
    )
    
    # Save visualization
    output_path = os.path.join(output_dir, f'demand_chart_{timestamp}.html')
    fig.write_html(output_path)
    
    # Also create sector chart
    sector_data = pd.DataFrame(demand_data['by_sector'])
    if not sector_data.empty:
        sector_fig = px.bar(
            sector_data,
            x='sector',
            y='job_count',
            color='avg_days_posted',
            color_continuous_scale='Viridis',
            labels={'sector': 'Sector', 'job_count': 'Number of Jobs', 'avg_days_posted': 'Avg Days Posted'},
            title='Job Demand by Sector'
        )
        
        sector_fig.update_layout(
            xaxis_title='Sector',
            yaxis_title='Number of Job Listings',
            height=500,
            width=900
        )
        
        sector_output_path = os.path.join(output_dir, f'sector_demand_chart_{timestamp}.html')
        sector_fig.write_html(sector_output_path)
    
    return output_path

def create_skill_chart(skill_data: Dict[str, Any], output_dir: str, timestamp: str) -> str:
    """
    Create a chart showing top skills and their wage correlations.
    
    Args:
        skill_data: Skill analysis data
        output_dir: Directory to save visualization
        timestamp: Timestamp string for filename
        
    Returns:
        Path to saved visualization
    """
    # Extract skills with wage data
    skills_wage_data = pd.DataFrame(skill_data['skills_with_wages'])
    if skills_wage_data.empty:
        return ""
    
    # Sort by average wage
    skills_wage_data = skills_wage_data.sort_values('avg_wage', ascending=False)
    top_wage_skills = skills_wage_data.head(10)
    
    # Create bar chart
    fig = px.bar(
        top_wage_skills,
        x='skill',
        y='avg_wage',
        color='job_count',
        color_continuous_scale='Viridis',
        labels={
            'skill': 'Skill', 
            'avg_wage': 'Average Hourly Wage ($)', 
            'job_count': 'Number of Jobs'
        },
        title='Top Skills by Average Wage'
    )
    
    fig.update_layout(
        xaxis_title='Skill',
        yaxis_title='Average Hourly Wage ($)',
        height=500,
        width=900,
        xaxis={'categoryorder':'total descending'}
    )
    
    # Save visualization
    output_path = os.path.join(output_dir, f'skill_wage_chart_{timestamp}.html')
    fig.write_html(output_path)
    
    # Create a frequency chart for most common skills
    top_skills = pd.DataFrame(skill_data['top_skills'])
    if not top_skills.empty:
        skill_freq_fig = px.bar(
            top_skills.head(15),
            x='count',
            y='skill',
            orientation='h',
            color='count',
            color_continuous_scale='Viridis',
            labels={'count': 'Frequency', 'skill': 'Skill'},
            title='Most Common Skills in Job Listings'
        )
        
        skill_freq_fig.update_layout(
            xaxis_title='Number of Job Listings',
            yaxis_title='Skill',
            height=600,
            width=900
        )
        
        skill_freq_path = os.path.join(output_dir, f'skill_frequency_chart_{timestamp}.html')
        skill_freq_fig.write_html(skill_freq_path)
    
    return output_path

def create_opportunity_chart(analysis_results: Dict[str, Any], output_dir: str, timestamp: str) -> str:
    """
    Create a bubble chart of market opportunities.
    
    Args:
        analysis_results: Complete analysis results
        output_dir: Directory to save visualization
        timestamp: Timestamp string for filename
        
    Returns:
        Path to saved visualization
    """
    # Check if we have top markets data
    if 'top_markets' not in analysis_results:
        return ""
    
    markets = pd.DataFrame(analysis_results['top_markets'])
    if markets.empty:
        return ""
    
    # Create bubble chart
    fig = px.scatter(
        markets,
        x='avg_wage',
        y='avg_days_posted',
        size='job_count',
        color='opportunity_score',
        hover_name='city',
        text='sector',
        labels={
            'avg_wage': 'Average Hourly Wage ($)',
            'avg_days_posted': 'Average Days Posted',
            'job_count': 'Number of Jobs',
            'opportunity_score': 'Opportunity Score'
        },
        title='Market Opportunities by City and Sector',
        size_max=50,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title='Average Hourly Wage ($)',
        yaxis_title='Average Days Posted (Higher = Harder to Fill)',
        height=600,
        width=900
    )
    
    fig.update_traces(
        textposition='top center',
        textfont=dict(size=10, color='black')
    )
    
    # Save visualization
    output_path = os.path.join(output_dir, f'opportunity_chart_{timestamp}.html')
    fig.write_html(output_path)
    
    return output_path

def generate_visualizations(analysis_results: Dict[str, Any], output_dir: str, timestamp: str) -> Dict[str, str]:
    """
    Generate all visualizations for the analysis results.
    
    Args:
        analysis_results: Complete analysis results
        output_dir: Directory to save visualizations
        timestamp: Timestamp string for filenames
        
    Returns:
        Dictionary mapping chart names to file paths
    """
    charts = {}
    
    # Wage heatmap
    wage_chart_path = create_wage_heatmap(
        analysis_results['wage_analysis'],
        output_dir,
        timestamp
    )
    if wage_chart_path:
        charts['wage_heatmap'] = wage_chart_path
    
    # Demand bar chart
    demand_chart_path = create_demand_bar_chart(
        analysis_results['demand_analysis'],
        output_dir,
        timestamp
    )
    if demand_chart_path:
        charts['demand_chart'] = demand_chart_path
    
    # Skill chart
    skill_chart_path = create_skill_chart(
        analysis_results['skill_analysis'],
        output_dir,
        timestamp
    )
    if skill_chart_path:
        charts['skill_chart'] = skill_chart_path
    
    # Opportunity chart
    opportunity_chart_path = create_opportunity_chart(
        analysis_results,
        output_dir,
        timestamp
    )
    if opportunity_chart_path:
        charts['opportunity_chart'] = opportunity_chart_path
    
    return charts

def _create_matplotlib_figure(fig_data: Any, fig_type: str) -> Figure:
    """
    Create a matplotlib figure for inclusion in the PDF report.
    
    Args:
        fig_data: Data to visualize
        fig_type: Type of figure to create
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    if fig_type == 'wage_by_city':
        # Create bar chart of wages by city
        cities = [city['city'] for city in fig_data['by_city'][:8]]
        wages = [city['mean'] for city in fig_data['by_city'][:8]]
        
        ax.bar(cities, wages, color='teal')
        ax.set_title('Average Hourly Wages by City')
        ax.set_xlabel('City')
        ax.set_ylabel('Average Hourly Wage ($)')
        ax.set_ylim(bottom=0)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')
        
    elif fig_type == 'wage_by_sector':
        # Create bar chart of wages by sector
        sectors = [sector['sector'] for sector in fig_data['by_sector']]
        wages = [sector['mean'] for sector in fig_data['by_sector']]
        
        ax.bar(sectors, wages, color='teal')
        ax.set_title('Average Hourly Wages by Sector')
        ax.set_xlabel('Sector')
        ax.set_ylabel('Average Hourly Wage ($)')
        ax.set_ylim(bottom=0)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')
            
    elif fig_type == 'demand_by_city':
        # Create horizontal bar chart of job counts by city
        cities = [city['city'] for city in fig_data['by_city'][:8]]
        job_counts = [city['job_count'] for city in fig_data['by_city'][:8]]
        avg_days = [city['avg_days_posted'] for city in fig_data['by_city'][:8]]
        
        bars = ax.barh(cities, job_counts, color='purple')
        ax.set_title('Job Demand by City')
        ax.set_xlabel('Number of Job Listings')
        ax.set_ylabel('City')
        
        # Add avg days as text
        for i, bar in enumerate(bars):
            ax.text(
                bar.get_width() + 1, 
                bar.get_y() + bar.get_height()/2, 
                f"{avg_days[i]:.1f} days", 
                va='center'
            )
            
    elif fig_type == 'skills_frequency':
        # Create horizontal bar chart of top skills
        if not fig_data['top_skills']:
            fig.text(0.5, 0.5, "No skill data available", ha='center', va='center')
        else:
            skills = [skill['skill'] for skill in fig_data['top_skills'][:10]]
            counts = [skill['count'] for skill in fig_data['top_skills'][:10]]
            
            y_pos = np.arange(len(skills))
            ax.barh(y_pos, counts, color='green')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(skills)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_title('Most Common Skills in Job Listings')
            ax.set_xlabel('Number of Listings')
    
    fig.tight_layout()
    return fig

def create_report(
    analysis_results: Dict[str, Any],
    chart_paths: Dict[str, str],
    skill_data: Dict[str, List[Tuple[str, int]]],
    output_path: str
) -> str:
    """
    Create a comprehensive PDF report with analysis results.
    
    Args:
        analysis_results: Complete analysis results
        chart_paths: Paths to generated chart visualizations
        skill_data: Key skills by sector
        output_path: Path to save the PDF report
        
    Returns:
        Path to saved PDF report
    """
    # Initialize PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Define custom styles
    title_style = styles['Title']
    heading_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Add title
    story.append(Paragraph("Job Market Analysis Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Spacer(1, 0.25 * inch))
    
    # Add executive summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    # Calculate top-level metrics for summary
    wage_data = analysis_results['wage_analysis']
    demand_data = analysis_results['demand_analysis']
    
    # Define summary table data
    summary_data = [
        ["Total Jobs Analyzed", str(analysis_results['job_count'])],
        ["Cities Covered", str(analysis_results['city_count'])],
        ["Sectors Analyzed", str(analysis_results['sector_count'])],
        ["Average Hourly Wage", f"${wage_data['overall']['mean']:.2f}"],
        ["Jobs With Posted Wages", f"{wage_data['overall'].get('pct_with_wage', 0):.1f}%"],
        ["Recently Posted Jobs (≤ 3 days)", str(demand_data['overall'].get('recent_postings', 0))],
        ["Average Days Posted", f"{demand_data['overall'].get('avg_days_posted', 0):.1f} days"],
    ]
    
    # Create summary table
    summary_table = Table(summary_data, colWidths=[3 * inch, 2 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.25 * inch))
    
    # Add key findings
    story.append(Paragraph("Key Findings", heading2_style))
    
    # Extract key insights
    key_findings = []
    
    # Wage insights
    if wage_data['by_sector']:
        highest_wage_sector = max(wage_data['by_sector'], key=lambda x: x['mean'])
        key_findings.append(
            f"• {highest_wage_sector['sector']} sector offers the highest average wage at "
            f"${highest_wage_sector['mean']:.2f} per hour"
        )
    
    if wage_data['by_city']:
        highest_wage_city = max(wage_data['by_city'], key=lambda x: x['mean'])
        key_findings.append(
            f"• {highest_wage_city['city']} has the highest average wages at "
            f"${highest_wage_city['mean']:.2f} per hour"
        )
    
    # Demand insights
    if demand_data['by_city']:
        highest_demand_city = max(demand_data['by_city'], key=lambda x: x['job_count'])
        longest_open_city = max(demand_data['by_city'], key=lambda x: x['avg_days_posted'])
        
        key_findings.append(
            f"• {highest_demand_city['city']} has the highest job demand with "
            f"{highest_demand_city['job_count']} open positions"
        )
        
        key_findings.append(
            f"• {longest_open_city['city']} has the longest-standing job openings, averaging "
            f"{longest_open_city['avg_days_posted']:.1f} days posted"
        )
    
    # Top opportunities
    if 'top_markets' in analysis_results and analysis_results['top_markets']:
        top_opportunity = analysis_results['top_markets'][0]
        key_findings.append(
            f"• Top market opportunity: {top_opportunity['sector']} in {top_opportunity['city']} "
            f"(${top_opportunity['avg_wage']:.2f}/hr, {top_opportunity['job_count']} jobs, "
            f"{top_opportunity['avg_days_posted']:.1f} days posted)"
        )
    
    # Add findings to report
    for finding in key_findings:
        story.append(Paragraph(finding, normal_style))
    
    story.append(Spacer(1, 0.25 * inch))
    
    # Add wage analysis section
    story.append(PageBreak())
    story.append(Paragraph("Wage Analysis", heading_style))
    
    # Add wage data visualization
    if wage_data['by_city'] and wage_data['by_sector']:
        # Create wage by city chart for PDF
        wage_city_fig = _create_matplotlib_figure(wage_data, 'wage_by_city')
        img_data = io.BytesIO()
        wage_city_fig.savefig(img_data, format='png', dpi=100)
        img_data.seek(0)
        wage_city_img = Image(img_data, width=6*inch, height=3*inch)
        story.append(wage_city_img)
        story.append(Spacer(1, 0.15 * inch))
        
        # Create wage by sector chart for PDF
        wage_sector_fig = _create_matplotlib_figure(wage_data, 'wage_by_sector')
        img_data = io.BytesIO()
        wage_sector_fig.savefig(img_data, format='png', dpi=100)
        img_data.seek(0)
        wage_sector_img = Image(img_data, width=6*inch, height=3*inch)
        story.append(wage_sector_img)
    else:
        story.append(Paragraph("No wage data available for visualization.", normal_style))
    
    # Add wage table
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Average Wages by Sector", heading2_style))
    
    if wage_data['by_sector']:
        wage_sector_data = [['Sector', 'Avg. Hourly Wage', 'Median', '# Jobs']]
        for sector in wage_data['by_sector']:
            wage_sector_data.append([
                sector['sector'],
                f"${sector['mean']:.2f}",
                f"${sector['median']:.2f}",
                str(sector['count'])
            ])
        
        wage_sector_table = Table(wage_sector_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1*inch])
        wage_sector_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ]))
        
        story.append(wage_sector_table)
    else:
        story.append(Paragraph("No sector wage data available.", normal_style))
    
    # Add demand analysis section
    story.append(PageBreak())
    story.append(Paragraph("Demand Analysis", heading_style))
    
    # Add demand visualization
    if demand_data['by_city']:
        # Create demand by city chart for PDF
        demand_fig = _create_matplotlib_figure(demand_data, 'demand_by_city')
        img_data = io.BytesIO()
        demand_fig.savefig(img_data, format='png', dpi=100)
        img_data.seek(0)
        demand_img = Image(img_data, width=6*inch, height=3.5*inch)
        story.append(demand_img)
    else:
        story.append(Paragraph("No demand data available for visualization.", normal_style))
    
    # Add top opportunities table
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Top Market Opportunities", heading2_style))
    
    if 'top_markets' in analysis_results and analysis_results['top_markets']:
        opportunities_data = [['City', 'Sector', 'Jobs', 'Avg. Days Posted', 'Avg. Wage']]
        
        for market in analysis_results['top_markets'][:10]:
            opportunities_data.append([
                market['city'],
                market['sector'],
                str(market['job_count']),
                f"{market['avg_days_posted']:.1f}",
                f"${market['avg_wage']:.2f}" if market['avg_wage'] else "N/A"
            ])
        
        opportunities_table = Table(opportunities_data, colWidths=[1.2*inch, 1.5*inch, 0.8*inch, 1.2*inch, 1*inch])
        opportunities_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (2, 1), (-1, -1), 'RIGHT'),
        ]))
        
        story.append(opportunities_table)
    else:
        story.append(Paragraph("No market opportunity data available.", normal_style))
    
    # Add skills analysis section
    story.append(PageBreak())
    story.append(Paragraph("Skills Analysis", heading_style))
    
    # Add skills visualization
    skill_analysis = analysis_results['skill_analysis']
    if skill_analysis:
        skills_fig = _create_matplotlib_figure(skill_analysis, 'skills_frequency')
        img_data = io.BytesIO()
        skills_fig.savefig(img_data, format='png', dpi=100)
        img_data.seek(0)
        skills_img = Image(img_data, width=6*inch, height=3.5*inch)
        story.append(skills_img)
    
    # Add skills by sector
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Top Skills by Sector", heading2_style))
    
    for sector, skills in skill_data.items():
        if skills:
            story.append(Paragraph(f"<b>{sector.title()}</b>", normal_style))
            
            skill_text = ", ".join([f"{skill[0]} ({skill[1]})" for skill in skills[:8]])
            story.append(Paragraph(skill_text, normal_style))
            story.append(Spacer(1, 0.1 * inch))
    
    # Add skills with highest wages
    if skill_analysis.get('skills_with_wages'):
        story.append(Spacer(1, 0.25 * inch))
        story.append(Paragraph("Skills with Highest Average Wages", heading2_style))
        
        wage_skills_data = [['Skill', 'Avg. Hourly Wage', '# Jobs']]
        
        for skill in skill_analysis['skills_with_wages'][:10]:
            wage_skills_data.append([
                skill['skill'],
                f"${skill['avg_wage']:.2f}",
                str(skill['job_count'])
            ])
        
        wage_skills_table = Table(wage_skills_data, colWidths=[2.5*inch, 2*inch, 1*inch])
        wage_skills_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ]))
        
        story.append(wage_skills_table)
    
    # Add conclusion
    story.append(PageBreak())
    story.append(Paragraph("Conclusions & Recommendations", heading_style))
    
    # Generate some insights for Traba based on the data
    recommendations = [
        "1. <b>High-Opportunity Markets:</b> Focus worker recruitment and business development in markets with "
        "high demand and longer posting times, as these represent gaps in current supply.",
        
        "2. <b>Skill Development:</b> Offer training or certification opportunities for workers in high-wage skills "
        "to increase their earning potential and marketability.",
        
        "3. <b>Competitive Wages:</b> Ensure Traba's rates are competitive with market averages identified in this "
        "analysis, particularly in high-demand sectors.",
        
        "4. <b>Market Expansion:</b> Consider entering markets with high opportunity scores first when "
        "planning geographic expansion.",
        
        "5. <b>Worker Retention:</b> Highlight Traba's fast job-filling capabilities (under 1 day) compared "
        "to the industry average found in this analysis."
    ]
    
    for rec in recommendations:
        story.append(Paragraph(rec, normal_style))
        story.append(Spacer(1, 0.1 * inch))
    
      # Add methodology note
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("Methodology Note", heading2_style))
    methodology = (
        "This analysis was performed using data collected from job boards for the specified cities and sectors. "
        "All wage data was normalized to hourly rates for consistent comparison. The job posting age was "
        "determined based on 'days ago' information provided by job listings. Skills were extracted using "
        "natural language processing techniques applied to job descriptions. Market opportunities were "
        "identified by analyzing a combination of job counts, wage levels, and average posting duration."
    )
    story.append(Paragraph(methodology, normal_style))
    
    # Build the PDF
    doc.build(story)
    
    return output_path
