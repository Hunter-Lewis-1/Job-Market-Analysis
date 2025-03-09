#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Collection Module for Sentiment Analysis Project

This module handles collection of news articles about specified companies using Google News.
"""

import re
import logging
import requests
import pandas as pd
from datetime import datetime
from time import sleep
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from googlenews import GoogleNews

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_article_content(session, url):
    """Fetch and extract the main content from an article URL"""
    try:
        response = session.get(url, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"HTTP {response.status_code} for {url}")
            return ""
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove noise elements
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.extract()
        
        # Prioritized content extraction strategy
        # 1. Try article tag first (most specific)
        article = soup.find('article')
        if article:
            paragraphs = article.find_all('p')
            if paragraphs:
                return ' '.join(p.get_text().strip() for p in paragraphs)
        
        # 2. Try common content div classes
        for div_class in ['content', 'article-content', 'post-content', 'entry-content', 'story-body']:
            content_div = soup.find('div', class_=re.compile(div_class))
            if content_div:
                paragraphs = content_div.find_all('p')
                if paragraphs:
                    return ' '.join(p.get_text().strip() for p in paragraphs)
        
        # 3. Fall back to all paragraphs in the body
        paragraphs = soup.find_all('p')
        if paragraphs:
            return ' '.join(p.get_text().strip() for p in paragraphs)
        
        # 4. Last resort: all text from the body
        body = soup.find('body')
        if body:
            text = body.get_text()
            return re.sub(r'\s+', ' ', text).strip()
            
        return ""
        
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return ""

def collect_news_articles(company, use_dummy=False):
    """
    Collect news articles for a specific company from Google News since 2021.

    Parameters:
    ----------
    company : str
        Company name to collect articles for
    use_dummy : bool
        If True, return dummy data for testing

    Returns:
    -------
    list
        List of articles with keys: title, text, url, publication_date, source
    """
    # Handle dummy data case
    if use_dummy:
        return [
            {
                'title': f'Dummy article about {company}',
                'text': f'This is a dummy article about {company} for testing purposes.',
                'url': 'https://example.com/dummy',
                'publication_date': '2023-01-01',
                'source': 'Dummy Source'
            },
            {
                'title': f'Another dummy article about {company}',
                'text': f'This is another dummy article about {company}.',
                'url': 'https://example.com/dummy2',
                'publication_date': '2023-02-01',
                'source': 'Dummy Source 2'
            }
        ]

    logger.info(f"Collecting news articles for {company} using Google News")
    
    # Create HTTP session for connection pooling
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    # Initialize GoogleNews with date range from 2021
    googlenews = GoogleNews(lang='en', region='US')
    googlenews.set_time_range('01/01/2021', datetime.now().strftime('%m/%d/%Y'))
    googlenews.search(company)
    
    articles = []
    processed_urls = set()

    # Process Google News results
    for page in range(1, 6):  # Fetch up to 5 pages for testing
        try:
            if page > 1:
                googlenews.get_page(page)
            results = googlenews.results()
            
            if not results:
                logger.warning(f"No results on page {page} for {company}")
                break
            
            for entry in results:
                try:
                    title = entry.get('title', '')
                    link = entry.get('link', '')
                    
                    if link in processed_urls:
                        continue
                    processed_urls.add(link)
                    
                    # Fetch full article content
                    content = get_article_content(session, link)
                    if not content or len(content) < 50:
                        continue
                    
                    # Get publication date
                    pub_date = entry.get('date', datetime.now().strftime('%Y-%m-%d'))
                    if isinstance(pub_date, str):
                        try:
                            pub_date = datetime.strptime(pub_date, '%Y-%m-%d').strftime('%Y-%m-%d')
                        except ValueError:
                            pub_date = datetime.now().strftime('%Y-%m-%d')
                    
                    article = {
                        'title': title,
                        'text': content,
                        'url': link,
                        'publication_date': pub_date,
                        'source': entry.get('media', 'Google News')
                    }
                    
                    articles.append(article)
                    logger.info(f"Collected article about {company} from Google News")
                    
                except Exception as e:
                    logger.error(f"Error processing entry for {company}: {str(e)}")
                    continue
                
                sleep(0.1)  # Small delay to avoid rate limiting
                
        except Exception as e:
            logger.error(f"Error fetching page {page} for {company}: {str(e)}")
            break
    
    # Sort by publication date (newest first)
    articles = sorted(
        articles,
        key=lambda x: x.get('publication_date', ''),
        reverse=True
    )
    
    logger.info(f"Total articles collected for {company}: {len(articles)}")
    return articles
