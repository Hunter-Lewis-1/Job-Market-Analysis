#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Company Context Filter
Eliminates false positives where a company name appears in text
but the context is unrelated to the actual company.
"""

import re
import spacy
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Load spaCy model - small model for efficiency
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class CompanyContextFilter:
    """Filter to determine if text genuinely refers to a specific company"""
    
    def __init__(self):
        # Company-specific context indicators (customize for your companies)
        self.company_contexts = {
            "Traba": {
                "industry_terms": ["staffing", "workforce", "gig economy", "labor", "workers", 
                                  "temporary staffing", "shifts", "marketplace", "platform"],
                "company_identifiers": ["Traba Inc", "Traba platform", "Traba app", "Traba.work"],
                "founders": ["Mike Shebat", "Akshay Buddiga"],
                "min_score": 0.6  # Minimum confidence score to consider relevant
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
        """
        Extract text surrounding company name mentions
        Uses regex with efficient word boundary matching
        
        Args:
            text: Full article text
            company_name: Name of company to find
            window: Number of characters before/after to include
        
        Returns:
            List of context snippets
        """
        pattern = r'\b' + re.escape(company_name) + r'\b'
        contexts = []
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            contexts.append(text[start:end])
            
        return contexts
    
    def analyze_entity_type(self, text, company_name):
        """
        Check if company name is recognized as an organization entity
        
        Args:
            text: Text to analyze
            company_name: Company name to check
        
        Returns:
            Boolean indicating if name is recognized as organization
        """
        # Process with spaCy for named entity recognition
        doc = nlp(text)
        
        # Check entities
        is_org = False
        for ent in doc.ents:
            if (ent.label_ == "ORG" and 
                company_name.lower() in ent.text.lower()):
                is_org = True
                break
                
        return is_org
    
    def contains_industry_terms(self, text, company_name):
        """
        Check if text contains industry-specific terms for the company
        Uses vectorized operations for efficiency
        
        Args:
            text: Text to analyze
            company_name: Company to check terms for
            
        Returns:
            Set of matched terms
        """
        text_lower = text.lower()
        terms = set()
        
        # Get terms specific to this company
        industry_terms = self.company_contexts.get(company_name, {}).get("industry_terms", [])
        identifiers = self.company_contexts.get(company_name, {}).get("company_identifiers", [])
        
        # Check industry terms
        for term in industry_terms:
            if term.lower() in text_lower:
                terms.add(term)
                
        # Check company identifiers (stronger signals)
        for identifier in identifiers:
            if identifier.lower() in text_lower:
                terms.add(identifier)
                
        return terms
    
    def is_about_company(self, text, company_name, threshold=None):
        """
        Determine if text is genuinely about the specified company
        
        Args:
            text: Article text to analyze
            company_name: Company name to check
            threshold: Optional custom threshold (overrides company default)
            
        Returns:
            (is_relevant, confidence_score, context_snippets)
        """
        if not text or not company_name:
            return False, 0.0, []
        
        # Get company-specific threshold
        if threshold is None:
            threshold = self.company_contexts.get(company_name, {}).get("min_score", 0.6)
            
        # Initialize feature scores
        scores = {
            "name_frequency": 0.0,      # How often company is mentioned
            "is_org_entity": 0.0,       # If recognized as organization
            "industry_terms": 0.0,      # Industry-specific terms present
            "company_identifiers": 0.0  # Company-specific identifiers
        }
        
        # Calculate normalized name frequency score (with diminishing returns)
        pattern = r'\b' + re.escape(company_name) + r'\b'
        mentions = len(re.findall(pattern, text, re.IGNORECASE))
        if mentions > 0:
            # Log scale to handle different article lengths
            scores["name_frequency"] = min(0.7, 0.2 + 0.5 * (np.log1p(mentions) / np.log1p(10)))
        else:
            # If no exact matches, it's definitely not relevant
            return False, 0.0, []
            
        # Get surrounding context for mentions
        contexts = self.get_surrounding_context(text, company_name)
        if not contexts:
            return False, 0.0, []
            
        # Check if recognizable as organization entity (in any context)
        for context in contexts[:min(3, len(contexts))]:  # Limit to first 3 for efficiency
            if self.analyze_entity_type(context, company_name):
                scores["is_org_entity"] = 0.7
                break
                
        # Check for industry terms and company identifiers
        terms = self.contains_industry_terms(text, company_name)
        
        industry_terms = set(self.company_contexts.get(company_name, {}).get("industry_terms", []))
        company_identifiers = set(self.company_contexts.get(company_name, {}).get("company_identifiers", []))
        
        # Score based on percentage of terms found (with higher weight for identifiers)
        if industry_terms:
            industry_matches = len([t for t in terms if t in industry_terms])
            scores["industry_terms"] = min(0.8, 0.8 * (industry_matches / len(industry_terms)))
            
        if company_identifiers:
            identifier_matches = len([t for t in terms if t in company_identifiers])
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
    
    def filter_articles(self, articles, company_name, parallel=True):
        """
        Filter a list of articles to keep only those genuinely about the company
        Uses parallel processing for large datasets
        
        Args:
            articles: List of article dictionaries with 'content' field
            company_name: Company name to check
            parallel: Whether to use parallel processing
            
        Returns:
            (filtered_articles, rejected_articles)
        """
        filtered = []
        rejected = []
        
        if parallel and len(articles) > 10:
            # Parallel processing for larger datasets
            with ThreadPoolExecutor(max_workers=min(8, len(articles))) as executor:
                # Create tasks for each article
                future_to_article = {
                    executor.submit(self.is_about_company, 
                                   article['content'], 
                                   company_name): article
                    for article in articles
                }
                
                # Process results as they complete
                for future in future_to_article:
                    article = future_to_article[future]
                    try:
                        is_relevant, score, contexts = future.result()
                        article['relevance_score'] = score
                        
                        if is_relevant:
                            filtered.append(article)
                        else:
                            rejected.append(article)
                    except Exception as e:
                        print(f"Error processing article: {e}")
                        rejected.append(article)
        else:
            # Sequential processing for smaller datasets
            for article in articles:
                try:
                    is_relevant, score, contexts = self.is_about_company(
                        article['content'], company_name
                    )
                    article['relevance_score'] = score
                    
                    if is_relevant:
                        filtered.append(article)
                    else:
                        rejected.append(article)
                except Exception as e:
                    print(f"Error processing article: {e}")
                    rejected.append(article)
                    
        return filtered, rejected


# Example usage
def filter_company_articles(articles, company_name):
    """
    Filter articles to include only those genuinely about the company
    
    Args:
        articles: List of article dictionaries
        company_name: Company name to verify
        
    Returns:
        List of relevant articles
    """
    context_filter = CompanyContextFilter()
    relevant_articles, rejected = context_filter.filter_articles(articles, company_name)
    
    print(f"Filtered {len(articles)} articles for {company_name}:")
    print(f"  - {len(relevant_articles)} relevant articles kept")
    print(f"  - {len(rejected)} false positives rejected")
    
    return relevant_articles
