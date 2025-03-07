#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSS Feed-based News Collector for Sentiment Analysis
"""

import os
import re
import logging
import requests
import feedparser
import pandas as pd
from datetime import datetime
from time import sleep
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Top business news sites with RSS feeds
RSS_FEEDS = [
    # Major Business News
    ("Wall Street Journal", "https://feeds.a.dj.com/rss/RSSBusinessNews.xml", False),
    ("Financial Times", "https://www.ft.com/rss/home", False),
    ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html", False),
    ("Reuters Business", "https://www.reutersagency.com/feed/?best-topics=business", False),
    ("Bloomberg", "https://www.bloomberg.com/markets/feeds/sitemap_news.xml", False),
    ("Business Insider", "https://www.businessinsider.com/rss", False),
    ("Forbes", "https://www.forbes.com/business/feed/", False),
    
    # Tech and Startups
    ("TechCrunch", "https://techcrunch.com/feed/", True),
    ("VentureBeat", "https://venturebeat.com/feed/", True),
    ("Wired Business", "https://www.wired.com/feed/category/business/latest/rss", False),
    
    # HR and Workforce News
    ("HR Dive", "https://www.hrdive.com/feeds/news/", False),
    ("Staffing Industry Analysts", "https://www2.staffingindustry.com/rss", False),
    ("HR Executive", "https://hrexecutive.com/feed/", True),
    
    # Additional Sources
    ("Fast Company", "https://www.fastcompany.com/latest/rss", False),
    ("Inc.com", "https://www.inc.com/rss", True),
    ("Entrepreneur", "https://www.entrepreneur.com/latest.rss", True),
]

class NewsCollector:
    """Collects news articles from RSS feeds about specified companies"""
    
    def __init__(self, companies, max_articles_per_company=100):
        """
        Initialize the RSS feed collector
        
        Parameters:
        ----------
        companies : list
            List of company names to collect news for
        max_articles_per_company : int
            Maximum articles to collect per company
        """
        self.companies = companies
        self.max_articles_per_company = max_articles_per_company
        self.collected_articles = {company: [] for company in companies}
        self.processed_urls = set()
        
    def is_relevant_article(self, title, description, company):
        """Check if article is relevant to the specified company"""
        pattern = r'\b{}\b'.format(re.escape(company))
        
        if re.search(pattern, title, re.IGNORECASE):
            return True
            
        if description and re.search(pattern, description, re.IGNORECASE):
            return True
            
        return False
    
    def get_article_content(self, url):
        """Fetch and extract the main content from an article URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return ""
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove noise elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.extract()
            
            # First try article tag
            article = soup.find('article')
            if article:
                paragraphs = article.find_all('p')
                if paragraphs:
                    return ' '.join([p.get_text().strip() for p in paragraphs])
            
            # Try common content divs
            for div_class in ['content', 'article-content', 'post-content', 'entry-content']:
                content_div = soup.find('div', class_=re.compile(div_class))
                if content_div:
                    paragraphs = content_div.find_all('p')
                    if paragraphs:
                        return ' '.join([p.get_text().strip() for p in paragraphs])
            
            # Fall back to all paragraphs in the body
            paragraphs = soup.find_all('p')
            if paragraphs:
                return ' '.join([p.get_text().strip() for p in paragraphs])
            
            return ""
            
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return ""
    
    def process_feed(self, source_name, feed_url, full_text):
        """Process a single RSS feed and collect relevant articles"""
        logger.info(f"Processing: {source_name}")
        articles = []
        
        try:
            feed = feedparser.parse(feed_url)
            
            if not feed.entries:
                logger.warning(f"No entries found in {source_name}")
                return []
            
            for entry in feed.entries:
                # Check each company
                for company in self.companies:
                    # Skip if we already have enough articles for this company
                    if len(self.collected_articles[company]) >= self.max_articles_per_company:
                        continue
                        
                    title = entry.get('title', '')
                    description = entry.get('summary', '')
                    link = entry.get('link', '')
                    
                    # Skip if already processed
                    if link in self.processed_urls:
                        continue
                    
                    self.processed_urls.add(link)
                    
                    # Check if article is relevant to this company
                    if self.is_relevant_article(title, description, company):
                        # Get full content
                        content = ""
                        if full_text:
                            content = entry.get('content', [{}])[0].get('value', '') if 'content' in entry else ''
                            
                        if not content or len(content) < 100:
                            content = self.get_article_content(link)
                        
                        if not content:
                            content = description
                            
                        # Get publication date
                        pub_date = None
                        if 'published_parsed' in entry and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        elif 'updated_parsed' in entry and entry.updated_parsed:
                            pub_date = datetime(*entry.updated_parsed[:6])
                        else:
                            pub_date = datetime.now()
                            
                        article = {
                            'title': title,
                            'content': content,
                            'url': link,
                            'date': pub_date.strftime('%Y-%m-%d'),
                            'source': source_name
                        }
                        
                        self.collected_articles[company].append(article)
                        articles.append((company, article))
                        
                        logger.info(f"Collected article about {company} from {source_name}")
                        
                # Add a small delay to be respectful
                sleep(0.1)
                        
        except Exception as e:
            logger.error(f"Error processing {source_name}: {str(e)}")
            
        return articles
    
    def collect_articles(self):
        """Collect articles from all configured RSS feeds"""
        all_articles = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_feed = {
                executor.submit(self.process_feed, name, url, full_text): name
                for name, url, full_text in RSS_FEEDS
            }
            
            for future in future_to_feed:
                name = future_to_feed[future]
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                    logger.info(f"Completed {name}, found {len(articles)} relevant articles")
                except Exception as e:
                    logger.error(f"Error with {name}: {str(e)}")
        
        # Return all collected articles organized by company
        return self.collected_articles

def collect_news_for_companies(companies, max_articles=100):
    """
    Main function to collect news about specified companies
    
    Parameters:
    ----------
    companies : list
        List of company names to collect news for
    max_articles : int
        Maximum number of articles to collect per company
        
    Returns:
    -------
    dict
        Dictionary with companies as keys and lists of articles as values
    """
    collector = NewsCollector(companies, max_articles)
    articles = collector.collect_articles()
    
    for company, company_articles in articles.items():
        logger.info(f"Collected {len(company_articles)} articles for {company}")
    
    return articles
