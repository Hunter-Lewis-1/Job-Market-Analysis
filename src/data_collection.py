"""
Data collection module for job board scraping
Uses asynchronous requests for high-performance data collection
"""
import re
import asyncio
import random
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm.asyncio import tqdm as async_tqdm

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for scraping
BASE_URL = "https://www.indeed.com"
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0',
]

# Rate limiting parameters
MIN_DELAY = 1.5  # Minimum delay between requests
MAX_DELAY = 3.0  # Maximum delay between requests
SEMAPHORE_LIMIT = 5  # Maximum concurrent requests

def extract_wage_info(salary_text: str) -> Dict[str, Any]:
    """Extract structured wage data from salary text."""
    if not salary_text or salary_text == "Not Listed":
        return {'min_wage': None, 'max_wage': None, 'avg_wage': None, 'wage_rate': None}
    
    # Find all dollar amounts in the text
    amounts = re.findall(r'\$(\d+(?:,\d+)*(?:\.\d+)?)', salary_text)
    amounts = [float(amount.replace(',', '')) for amount in amounts if amount]
    
    # Determine the rate (hourly, yearly, etc.)
    rate = None
    if re.search(r'hour|hr', salary_text, re.I):
        rate = "hourly"
    elif re.search(r'year|annual', salary_text, re.I):
        rate = "annual"
    elif re.search(r'month', salary_text, re.I):
        rate = "monthly"
    elif re.search(r'week', salary_text, re.I):
        rate = "weekly"
    elif re.search(r'day', salary_text, re.I):
        rate = "daily"
    
    # Calculate min, max and avg wage
    if amounts:
        min_wage = min(amounts)
        max_wage = max(amounts)
        avg_wage = sum(amounts) / len(amounts)
    else:
        min_wage = max_wage = avg_wage = None
    
    return {
        'min_wage': min_wage,
        'max_wage': max_wage,
        'avg_wage': avg_wage,
        'wage_rate': rate
    }

def parse_posting_date(date_text: str) -> Dict[str, Any]:
    """Convert date text like '30+ days ago' to structured date info."""
    if not date_text:
        return {'days_ago': 0, 'posting_date': datetime.now().strftime('%Y-%m-%d')}
    
    # Match various date patterns
    if "today" in date_text.lower() or "just posted" in date_text.lower():
        days_ago = 0
    elif "yesterday" in date_text.lower():
        days_ago = 1
    elif re.search(r'(\d+)\+?\s*day', date_text.lower()):
        days_match = re.search(r'(\d+)\+?\s*day', date_text.lower())
        days_ago = int(days_match.group(1))
    elif re.search(r'(\d+)\+?\s*hour', date_text.lower()) or re.search(r'(\d+)\+?\s*minute', date_text.lower()):
        days_ago = 0
    else:
        days_ago = 0
    
    posting_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
    
    return {
        'days_ago': days_ago,
        'posting_date': posting_date
    }

def extract_job_details(job_card: BeautifulSoup) -> Optional[Dict[str, Any]]:
    """Extract structured job data from a BeautifulSoup job card element."""
    try:
        # Extract job title
        title_elem = job_card.select_one('h2.jobTitle, a.jcs-JobTitle, h2.jobCardShelfContainer a')
        if not title_elem:
            return None
        title = title_elem.get_text(strip=True)
        
        # Extract company name
        company_elem = job_card.select_one('span.companyName, a.companyName, div.company_location span.companyName')
        company = company_elem.get_text(strip=True) if company_elem else "Unknown"
        
        # Extract location
        location_elem = job_card.select_one('div.companyLocation')
        location = location_elem.get_text(strip=True) if location_elem else "Unknown"
        
        # Extract salary if available
        salary_elem = job_card.find(string=re.compile(r'\$[\d,\.]+'))
        if not salary_elem:
            salary_elem = job_card.select_one('div.salary-snippet-container')
        salary_text = salary_elem.get_text(strip=True) if salary_elem else "Not Listed"
        
        # Parse salary info
        salary_info = extract_wage_info(salary_text)
        
        # Extract job URL
        url_elem = job_card.select_one('a[id^="job_"], a[data-jk], h2.jobCardShelfContainer a')
        job_url = None
        if url_elem and 'href' in url_elem.attrs:
            job_url = urljoin(BASE_URL, url_elem['href'])
        
        # Extract posting date
        date_elem = job_card.select_one('span.date, div.result-footer span.date')
        date_text = date_elem.get_text(strip=True) if date_elem else "Today"
        date_info = parse_posting_date(date_text)
        
        # Extract job description snippet
        desc_elem = job_card.select_one('div.job-snippet, div.jobCardShelfContainer, div[class*="summary"]')
        description = desc_elem.get_text(strip=True) if desc_elem else ""
        
        # Extract job ID
        job_id = None
        if url_elem and 'href' in url_elem.attrs:
            id_match = re.search(r'jk=([^&]+)', url_elem['href'])
            if id_match:
                job_id = id_match.group(1)
            elif 'data-jk' in url_elem.attrs:
                job_id = url_elem['data-jk']
        
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

def get_request_headers() -> Dict[str, str]:
    """Generate request headers with a random user agent."""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }

async def fetch_jobs_async(session: aiohttp.ClientSession, 
                          city: str, 
                          sector: str, 
                          max_jobs: int, 
                          semaphore: asyncio.Semaphore) -> List[Dict[str, Any]]:
    """
    Asynchronously fetch jobs for a city and sector.
    
    Args:
        session: aiohttp ClientSession
        city: City to search
        sector: Job sector
        max_jobs: Maximum jobs to collect
        semaphore: Rate limiting semaphore
        
    Returns:
        List of job dictionaries
    """
    jobs = []
    start = 0
    jobs_per_page = 15
    
    while len(jobs) < max_jobs:
        # Use semaphore to limit concurrent requests
        async with semaphore:
            try:
                # Add randomized delay
                await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                
                params = {
                    'q': sector,
                    'l': city,
                    'sort': 'date',
                    'start': start
                }
                
                # Perform the async request
                async with session.get(f"{BASE_URL}/jobs", params=params, timeout=15) as response:
                    if response.status != 200:
                        logger.warning(f"HTTP {response.status} for {sector} in {city}")
                        break
                    
                    html_content = await response.text()
                    
                    # Check for CAPTCHA detection
                    if "captcha" in html_content.lower() or "security check" in html_content.lower():
                        logger.warning(f"CAPTCHA detected for {city}/{sector}, stopping")
                        break
                    
                    # Parse HTML
                    soup = BeautifulSoup(html_content, 'html.parser')
                    job_cards = soup.select('div.job_seen_beacon, div.result')
                    
                    if not job_cards:
                        break
                    
                    # Process job cards
                    for card in job_cards:
                        if len(jobs) >= max_jobs:
                            break
                        
                        job_data = extract_job_details(card)
                        if job_data:
                            job_data['sector'] = sector
                            job_data['search_city'] = city
                            jobs.append(job_data)
                    
                    # If we got fewer jobs than expected, we've hit the end
                    if len(job_cards) < jobs_per_page:
                        break
                    
                    # Move to next page
                    start += jobs_per_page
                
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                logger.error(f"Network error for {sector} in {city}: {str(e)}")
                break
            except Exception as e:
                logger.error(f"Error scraping {sector} in {city}: {str(e)}")
                break
    
    return jobs

async def collect_jobs_async(cities: List[str], sectors: List[str], max_jobs: int = 25) -> List[Dict[str, Any]]:
    """
    Collect jobs from multiple cities and sectors asynchronously.
    
    Args:
        cities: List of cities
        sectors: List of sectors
        max_jobs: Maximum jobs per city/sector
        
    Returns:
        List of all job data dictionaries
    """
    all_jobs = []
    
    # Create tasks for all city/sector combinations
    tasks = []
    for city in cities:
        for sector in sectors:
            tasks.append((city, sector))
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)
    
    # Connector with connection pooling
    connector = aiohttp.TCPConnector(
        limit=SEMAPHORE_LIMIT,
        ssl=False,
        ttl_dns_cache=300
    )
    
    # Create ClientSession with timeout
    timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_connect=10, sock_read=20)
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers=get_request_headers()
    ) as session:
        # Create tasks
        fetch_tasks = [
            fetch_jobs_async(session, city, sector, max_jobs, semaphore)
            for city, sector in tasks
        ]
        
        # Execute all tasks with progress bar
        results = await async_tqdm.gather(*fetch_tasks, desc="Collecting jobs")
        
        # Combine results
        for jobs in results:
            all_jobs.extend(jobs)
    
    return all_jobs
            time.sleep(random.uniform(2.0, 5.0))
        except Exception as e:
            logger.error(f"Error collecting {sector} jobs in {city}: {str(e)}")
    
    return all_jobs
