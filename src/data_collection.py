#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSS Feed-based News Collector with Context Filtering
Includes context filtering to ensure articles are genuinely about target companies.
"""

import os
import re
import logging
import requests
import feedparser
import pandas as pd
import numpy as np
import spacy
from datetime import datetime
from time import sleep
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spaCy model - small model for efficiency
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Top business news sites with RSS feeds (name, url, has_full_text)
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
    
    # Specialized HR Tech and Staffing
    ("The Staffing Stream", "https://www.thestaffingstream.com/feed/", True),
    ("Staffing Hub", "https://staffinghub.com/feed/", True),
    ("Recruiting Daily", "https://recruitingdaily.com/feed/", True),
    ("ERE.net", "https://www.ere.net/feed/", True),
]

class CompanyContextFilter:
    """Filter to determine if text genuinely refers to a specific company"""
    
    def __init__(self):
        # Company-specific context indicators
        self.company_contexts = {
            "Traba": {
                "industry_terms": ["staffing", "workforce", "gig economy", "labor", "workers", 
                                  "temporary staffing", "shifts", "marketplace", "platform"],
                "company_identifiers": ["Traba Inc", "Traba platform", "Traba app", "Traba.work"],
                "founders": ["Mike Shebat", "Akshay Buddiga"],
                "min_score": 0.6
            },
            "Instawork": {
                "industry_terms": ["staffing", "gig", "hospitality", "shifts", "workers", 
                                  "flexible work", "hourly work", "marketplace"],
                "company_identifiers": ["Instawork platform", "Instawork app"],
                "min_score": 0.6
            },
            "Wonolo": {
                "industry_terms": ["staffing", "on-demand", "flexible work", "gig economy", 
                                  "warehouse", "fulfillment", "frontline workers"],
                "company_identifiers": ["Wonolo platform", "Wonolo app", "Wonolo Inc"],
                "min_score": 0.6
            },
            "Ubeya": {
                "industry_terms": ["staffing", "workforce management", "hospitality", "event staffing",
                                  "staff management", "scheduling"],
                "company_identifiers": ["Ubeya platform", "Ubeya software", "Ubeya app"],
                "min_score": 0.6
            },
            "Sidekicker": {
                "industry_terms": ["staffing", "temp agency", "casual staff", "hospitality", 
                                  "event staff", "Australia", "New Zealand"],
                "company_identifiers": ["Sidekicker platform", "Sidekicker app"],
                "min_score": 0.6
            },
            "Zenjob": {
                "industry_terms": ["temporary staffing", "temp work", "student jobs", "Germany",
                                  "flexible jobs", "digital staffing"],
                "company_identifiers": ["Zenjob GmbH", "Zenjob app", "Zenjob platform"],
                "min_score": 0.6
            },
            "Veryable": {
                "industry_terms": ["on-demand labor", "manufacturing", "logistics", "operations",
                                  "flexible workforce", "industrial"],
                "company_identifiers": ["Veryable platform", "Veryable marketplace", "Veryable Inc"],
                "min_score": 0.6
            },
            "Shiftgig": {
                "industry_terms": ["staffing", "deployment", "workforce", "shifts", 
                                  "gig economy", "staffing agencies"],
                "company_identifiers": ["Shiftgig platform", "Deploy by Shiftgig"],
                "min_score": 0.6
            }
        }
    
    def get_surrounding_context(self, text, company_name, window=50):
        """Extract text surrounding company name mentions"""
        pattern = r'\b' + re.escape(company_name) + r'\b'
        contexts = []
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            contexts.append(text[start:end])
            
        return contexts
    
    def analyze_entity_type(self, text, company_name):
        """Check if company name is recognized as an organization entity"""
        # Process with spaCy for named entity recognition
        doc = nlp(text)
        
        # Check entities
        company_lower = company_name.lower()
        return any(ent.label_ == "ORG" and company_lower in ent.text.lower() for ent in doc.ents)
    
    def contains_industry_terms(self, text, company_name):
        """Check if text contains industry-specific terms for the company"""
        text_lower = text.lower()
        terms = set()
        
        # Get terms specific to this company
        industry_terms = self.company_contexts.get(company_name, {}).get("industry_terms", [])
        identifiers = self.company_contexts.get(company_name, {}).get("company_identifiers", [])
        
        # Check industry terms - vectorized operations for efficiency
        terms.update(term for term in industry_terms if term.lower() in text_lower)
        
        # Check company identifiers (stronger signals)
        terms.update(identifier for identifier in identifiers if identifier.lower() in text_lower)
                
        return terms
    
    def is_about_company(self, text, company_name, threshold=None):
        """Determine if text is genuinely about the specified company"""
        if not text or not company_name:
            return False, 0.0, []
        
        # Get company-specific threshold
        if threshold is None:
            threshold = self.company_contexts.get(company_name, {}).get("min_score", 0.6)
            
        # Initialize feature scores
        scores = {
            "name_frequency": 0.0,
            "is_org_entity": 0.0,
            "industry_terms": 0.0,
            "company_identifiers": 0.0
        }
        
        # Calculate normalized name frequency score (with diminishing returns)
        pattern = r'\b' + re.escape(company_name) + r'\b'
        mentions = len(re.findall(pattern, text, re.IGNORECASE))
        if mentions == 0:
            return False, 0.0, []
            
        scores["name_frequency"] = min(0.7, 0.2 + 0.5 * (np.log1p(mentions) / np.log1p(10)))
            
        # Get surrounding context for mentions (more efficient than full doc analysis)
        contexts = self.get_surrounding_context(text, company_name)
        if not contexts:
            return False, 0.0, []
            
        # Check if recognizable as organization entity
        for context in contexts[:min(3, len(contexts))]:  # Limit to first 3 for efficiency
            if self.analyze_entity_type(context, company_name):
                scores["is_org_entity"] = 0.7
                break
                
        # Check for industry terms and company identifiers
        terms = self.contains_industry_terms(text, company_name)
        
        industry_terms = set(self.company_contexts.get(company_name, {}).get("industry_terms", []))
        company_identifiers = set(self.company_contexts.get(company_name, {}).get("company_identifiers", []))
        
        # Score based on percentage of terms found
        if industry_terms:
            industry_matches = sum(1 for t in terms if t in industry_terms)
            scores["industry_terms"] = min(0.8, 0.8 * (industry_matches / len(industry_terms)))
            
        if company_identifiers:
            identifier_matches = sum(1 for t in terms if t in company_identifiers)
            if identifier_matches > 0:
                scores["company_identifiers"] = min(1.0, 0.5 + 0.5 * (identifier_matches / len(company_identifiers)))
                
        # Calculate final weighted score
        weights = {
            "name_frequency": 0.15,
            "is_org_entity": 0.25,
            "industry_terms": 0.3,
            "company_identifiers": 0.3
        }
        
        confidence_score = sum(score * weights[key] for key, score in scores.items())
        
        return confidence_score >= threshold, confidence_score, contexts


class NewsCollector:
    """Collects news articles from RSS feeds about specified companies with context filtering"""
    
    def __init__(self, companies, max_articles_per_company=100):
        """Initialize the RSS feed collector"""
        self.companies = companies
        self.max_articles_per_company = max_articles_per_company
        self.collected_articles = {company: [] for company in companies}
        self.processed_urls = set()
        self.context_filter = CompanyContextFilter()
        
        # Use request session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def is_relevant_article(self, title, description, company):
        """Check if article potentially mentions the company"""
        pattern = r'\b' + re.escape(company) + r'\b'
        
        if re.search(pattern, title, re.IGNORECASE):
            return True
            
        if description and re.search(pattern, description, re.IGNORECASE):
            return True
            
        return False
    
    def get_article_content(self, url):
        """Fetch and extract the main content from an article URL"""
        try:
            # Use existing session for connection pooling
            response = self.session.get(url, timeout=10)
            
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
    
    def process_feed(self, source_name, feed_url, full_text):
        """Process a single RSS feed and collect relevant articles"""
        logger.info(f"Processing: {source_name}")
        articles = []
        
        try:
            feed = feedparser.parse(feed_url)
            
            if not feed.entries:
                logger.warning(f"No entries found in {source_name}")
                return []
            
            # Pre-filter entries to avoid processing irrelevant ones
            relevant_entries = []
            for entry in feed.entries:
                title = entry.get('title', '')
                description = entry.get('summary', '')
                link = entry.get('link', '')
                
                # Skip if already processed
                if link in self.processed_urls:
                    continue
                    
                self.processed_urls.add(link)
                
                # Check relevance for any company (quick filter)
                for company in self.companies:
                    if self.is_relevant_article(title, description, company):
                        relevant_entries.append((entry, company))
                        break
            
            # Process only relevant entries (more detailed analysis)
            for entry, company in relevant_entries:
                title = entry.get('title', '')
                description = entry.get('summary', '')
                link = entry.get('link', '')
                
                # Skip if we already have enough articles for this company
                if len(self.collected_articles[company]) >= self.max_articles_per_company:
                    continue
                    
                # Get full content
                content = ""
                if full_text:
                    content = entry.get('content', [{}])[0].get('value', '') if 'content' in entry else ''
                    
                if not content or len(content) < 100:
                    content = self.get_article_content(link)
                
                if not content:
                    content = description
                
                # Skip if no meaningful content
                if not content or len(content) < 50:
                    continue
                    
                # Apply context filtering to verify article is about the company
                is_relevant, score, _ = self.context_filter.is_about_company(content, company)
                
                if not is_relevant:
                    logger.info(f"Filtered out article about {company}, relevance score: {score:.2f}")
                    continue
                    
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
                    'source': source_name,
                    'relevance_score': score
                }
                
                self.collected_articles[company].append(article)
                articles.append((company, article))
                
                logger.info(f"Collected article about {company} from {source_name}, relevance: {score:.2f}")
                
            # Add a small delay to be respectful
            sleep(0.1)
                        
        except Exception as e:
            logger.error(f"Error processing {source_name}: {str(e)}")
            
        return articles
    
    def collect_articles(self):
        """Collect articles from all configured RSS feeds"""
        logger.info(f"Starting RSS feed collection for: {', '.join(self.companies)}")
        
        max_workers = min(len(RSS_FEEDS), 8)  # Cap at 8 workers to avoid overwhelming servers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_feed = {
                executor.submit(self.process_feed, name, url, full_text): name
                for name, url, full_text in RSS_FEEDS
            }
            
            for future in future_to_feed:
                name = future_to_feed[future]
                try:
                    articles = future.result()
                    logger.info(f"Completed {name}, found {len(articles)} relevant articles")
                except Exception as e:
                    logger.error(f"Error with {name}: {str(e)}")
        
        # Print collection statistics
        for company, articles in self.collected_articles.items():
            logger.info(f"Collected {len(articles)} verified articles for {company}")
        
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
    return collector.collect_articles()


def get_news_for_company(company_name, max_articles=20):
    """
    Collect news for a single company - convenience function for external calls
    
    Parameters:
    ----------
    company_name : str
        Company name to collect news for
    max_articles : int
        Maximum number of articles to collect
        
    Returns:
    -------
    list
        List of articles
    """
    results = collect_news_for_companies([company_name], max_articles)
    return results.get(company_name, [])


if __name__ == "__main__":
    # Example usage
    companies = ["Traba", "Instawork", "Wonolo", "Ubeya", "Sidekicker", "Zenjob", "Veryable", "Shiftgig"]
    articles = collect_news_for_companies(companies, max_articles=20)
    
    # Print summary
    print("\nCollection Summary:")
    for company, company_articles in articles.items():
        print(f"{company}: {len(company_articles)} articles")
