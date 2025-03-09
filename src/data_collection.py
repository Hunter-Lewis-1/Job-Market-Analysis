"""
Data collection module for scraping job listings from Indeed.
Uses rotating user agents.
"""
import logging
import time
import random
import re
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://www.indeed.com"
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
]

def create_session() -> requests.Session:
    """Create a requests session with a random user agent."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'https://www.google.com/'
    })
    return session

def parse_salary(salary_text: str) -> Dict[str, Any]:
    """
    Extract salary information from text.
    
    Args:
        salary_text: String containing salary information
        
    Returns:
        Dictionary with min_wage, max_wage, and wage_rate
    """
    if not salary_text or salary_text == "Not Listed":
        return {'min_wage': None, 'max_wage': None, 'avg_wage': None, 'wage_rate': None}
    
    # Find all numbers in the text
    amounts = re.findall(r'\$(\d+(?:\.\d+)?)', salary_text)
    
    # Determine the rate (hourly, daily, annual)
    rate = None
    if "hour" in salary_text.lower():
        rate = "hourly"
    elif "day" in salary_text.lower():
        rate = "daily"
    elif "year" in salary_text.lower() or "annual" in salary_text.lower():
        rate = "annual"
    elif "month" in salary_text.lower():
        rate = "monthly"
    elif "week" in salary_text.lower():
        rate = "weekly"
    
    # Calculate min, max and avg wage
    if amounts:
        amounts = [float(amount) for amount in amounts]
        min_wage = min(amounts)
        max_wage = max(amounts)
        avg_wage = sum(amounts) / len(amounts)
    else:
        min_wage = None
        max_wage = None
        avg_wage = None
    
    return {
        'min_wage': min_wage,
        'max_wage': max_wage, 
        'avg_wage': avg_wage,
        'wage_rate': rate
    }

def parse_posting_date(date_text: str) -> Dict[str, Any]:
    """
    Extract posting date information.
    
    Args:
        date_text: String containing date information like "30+ days ago"
        
    Returns:
        Dictionary with days_ago and posting_date
    """
    if not date_text:
        return {'days_ago': 0, 'posting_date': datetime.now().strftime('%Y-%m-%d')}
    
    # Try to extract the number of days
    days_match = re.search(r'(\d+)\+?\s*day', date_text.lower())
    if days_match:
        days_ago = int(days_match.group(1))
    elif "today" in date_text.lower():
        days_ago = 0
    elif "just posted" in date_text.lower():
        days_ago = 0
    elif "yesterday" in date_text.lower():
        days_ago = 1
    else:
        # If we can't determine, assume it's recent
        days_ago = 0
    
    posting_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
    
    return {
        'days_ago': days_ago,
        'posting_date': posting_date
    }

def extract_job_details(job_card: BeautifulSoup) -> Optional[Dict[str, Any]]:
    """
    Extract job details from a single job card.
    
    Args:
        job_card: BeautifulSoup object representing a job listing
        
    Returns:
        Dictionary with job details or None if parsing failed
    """
    try:
        # Extract job title
        title_elem = job_card.select_one('h2.jobTitle, a.jcs-JobTitle')
        if not title_elem:
            return None
        title = title_elem.get_text(strip=True)
        
        # Extract company name
        company_elem = job_card.select_one('span.companyName, a.companyName')
        company = company_elem.get_text(strip=True) if company_elem else "Unknown"
        
        # Extract location
        location_elem = job_card.select_one('div.companyLocation')
        location = location_elem.get_text(strip=True) if location_elem else "Unknown"
        
        # Extract salary if available
        salary_elem = job_card.find(string=re.compile(r'\$[\d,\.]+'))
        salary_text = salary_elem.strip() if salary_elem else "Not Listed"
        salary_info = parse_salary(salary_text)
        
        # Extract job URL
        url_elem = job_card.select_one('a[id^="job_"]')
        job_url = urljoin(BASE_URL, url_elem['href']) if url_elem and 'href' in url_elem.attrs else None
        
        # Extract posting date
        date_elem = job_card.select_one('span.date')
        date_text = date_elem.get_text(strip=True) if date_elem else "Today"
        date_info = parse_posting_date(date_text)
        
        # Extract job description snippet
        desc_elem = job_card.select_one('div.job-snippet, div.jobCardShelfContainer')
        description = desc_elem.get_text(strip=True) if desc_elem else ""
        
        # Extract job ID
        job_id = None
        if url_elem and 'href' in url_elem.attrs:
            id_match = re.search(r'jk=([^&]+)', url_elem['href'])
            if id_match:
                job_id = id_match.group(1)
        
        return {
            'job_id': job_id,
            'title': title,
            'company': company,
            'location': location,
            'salary_text': salary_text,
            'min_wage': salary_info['min_wage'],
            'max_wage': salary_info['max_wage'],
            'avg_wage': salary_info['avg_wage'],
            'wage_rate': salary_info['wage_rate'],
            'days_ago': date_info['days_ago'],
            'posting_date': date_info['posting_date'],
            'description': description,
            'url': job_url,
            'scrape_date': datetime.now().strftime('%Y-%m-%d')
        }
    except Exception as e:
        logger.error(f"Error extracting job details: {str(e)}")
        return None

def scrape_indeed_jobs(city: str, sector: str, max_jobs: int = 25) -> List[Dict[str, Any]]:
    """
    Scrape Indeed jobs for a specific city and sector.
    
    Args:
        city: City to search for jobs
        sector: Job sector to search
        max_jobs: Maximum number of jobs to collect per city/sector
        
    Returns:
        List of job listings as dictionaries
    """
    session = create_session()
    jobs = []
    start = 0
    jobs_per_page = 10
    
    while len(jobs) < max_jobs:
        # Construct the URL with query parameters
        params = {
            'q': sector,
            'l': city,
            'sort': 'date',
            'start': start
        }
        
        try:
            # Add jitter to avoid detection
            time.sleep(random.uniform(1.0, 3.0))
            
            response = session.get(
                f"{BASE_URL}/jobs", 
                params=params, 
                timeout=15
            )
            
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} for {sector} in {city}")
                break
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check for CAPTCHA
            if "captcha" in response.text.lower() or soup.find(string=re.compile(r'security\s+check', re.I)):
                logger.warning("CAPTCHA detected, stopping scrape")
                break
            
            # Find all job cards
            job_cards = soup.select('div.job_seen_beacon, div.result')
            
            if not job_cards:
                logger.info(f"No more job cards found for {sector} in {city}")
                break
            
            # Process each job card
            for card in job_cards:
                if len(jobs) >= max_jobs:
                    break
                
                job_data = extract_job_details(card)
                if job_data:
                    job_data['sector'] = sector
                    job_data['search_city'] = city
                    jobs.append(job_data)
            
            # If we didn't get full results, we've likely hit the end
            if len(job_cards) < jobs_per_page:
                break
            
            # Move to the next page
            start += jobs_per_page
            
        except Exception as e:
            logger.error(f"Error scraping {sector} in {city}: {str(e)}")
            break
    
    logger.info(f"Collected {len(jobs)} jobs for {sector} in {city}")
    return jobs

def collect_job_data(city: str, sectors: List[str], max_jobs_per_sector: int = 25) -> List[Dict[str, Any]]:
    """
    Collect job data for multiple sectors in a city.
    
    Args:
        city: City to search for jobs
        sectors: List of job sectors to search
        max_jobs_per_sector: Maximum number of jobs to collect per sector
        
    Returns:
        List of all jobs collected
    """
    all_jobs = []
    
    for sector in sectors:
        try:
            sector_jobs = scrape_indeed_jobs(city, sector, max_jobs_per_sector)
            all_jobs.extend(sector_jobs)
            # Brief pause between sectors
            time.sleep(random.uniform(2.0, 5.0))
        except Exception as e:
            logger.error(f"Error collecting {sector} jobs in {city}: {str(e)}")
    
    return all_jobs
