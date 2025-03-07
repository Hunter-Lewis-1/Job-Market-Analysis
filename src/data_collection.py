import requests
from bs4 import BeautifulSoup
import time
import logging
from urllib.parse import urljoin, quote

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_page(url, headers, max_retries=3, timeout=15):
    """Fetch page content with retries."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response.text
            logging.error(f"HTTP {response.status_code} for {url}")
        except Exception as e:
            logging.error(f"Failed to fetch {url}, attempt {attempt + 1}: {e}")
            time.sleep(2)
    return None

def extract_text_with_bs4(html, url):
    """Extract text from HTML using BeautifulSoup."""
    if not html:
        return ''
    try:
        soup = BeautifulSoup(html, 'html.parser')
        # Remove scripts and styles
        for script in soup(['script', 'style']):
            script.decompose()
        # Extract text from <p>, <article>, or main content areas
        content = soup.find('article') or soup.find('div', class_=['content', 'article', 'post']) or soup.body
        if content:
            paragraphs = content.find_all('p')
            text = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            if text:
                return text
        logging.warning(f"No meaningful text found in {url}")
        return ''
    except Exception as e:
        logging.error(f"Error extracting text from {url}: {e}")
        return ''

def is_relevant(text, title, url, query):
    """Score relevance based on company name and industry keywords."""
    relevance_keywords = [
        'staffing', 'gig economy', 'temporary workers', 'on-demand labor', 
        'workforce platform', 'hiring', 'jobs', 'labor', 'warehouse', 
        'distribution center', 'manufacturing', 'supply chain'
    ]
    query_lower = query.lower()
    content = (text.lower() + ' ' + title.lower() + ' ' + url.lower())
    company_mentioned = query_lower in content
    keyword_score = sum(1 for keyword in relevance_keywords if keyword in content)
    # Relevant if company name is present or at least 2 industry keywords
    return company_mentioned or keyword_score >= 2

def collect_news_articles(query, use_dummy=False):
    """
    Collects news articles by scraping targeted sources for a given company.
    
    Args:
        query (str): Company name (e.g., 'Traba', 'Instawork').
        use_dummy (bool): If True, returns dummy data.
    
    Returns:
        list: List of dicts with 'title', 'text', 'publication_date', 'url'.
    """
    articles_data = []
    
    if use_dummy:
        logging.info(f"Using dummy data for {query}")
        return [
            {'title': f"{query} Raises Funds", 'text': f"{query} secures $20M for staffing innovation.",
             'publication_date': '2025-03-01', 'url': 'http://example.com/dummy1'},
            {'title': f"{query} Expands", 'text': f"{query} grows operations in NY.",
             'publication_date': '2025-03-02', 'url': 'http://example.com/dummy2'}
        ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/'
    }
    
    # Dynamic source templates for each company
    query_encoded = quote(query.lower())
    base_urls = [
        f'https://www.crunchbase.com/organization/{query_encoded}',  # Company-specific Crunchbase page
        f'https://www.linkedin.com/company/{query_encoded}/posts/',  # LinkedIn posts
        f'https://www.bizjournals.com/search?q={query_encoded}',     # Business Journals search
        f'https://news.google.com/search?q={query}+staffing+-inurl:({query}.com+{query}.work)'  # Google News with exclusions
    ]
    
    all_articles = []
    seen_urls = set()
    
    for base_url in base_urls:
        logging.info(f"Scraping {base_url} for {query}")
        html = fetch_page(base_url, headers)
        if not html:
            continue
        
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.find_all('a', href=True)
        
        for link in links[:50]:  # Limit to 50 links per source
            title = link.get_text(strip=True) or 'No title'
            url = urljoin(base_url, link['href'])
            if url in seen_urls or any(domain in url for domain in [f'{query.lower()}.com', f'{query.lower()}.work']):
                continue
            
            article_html = fetch_page(url, headers)
            text = extract_text_with_bs4(article_html, url)
            if not text:
                continue
            
            if is_relevant(text, title, url, query):
                all_articles.append({
                    'title': title,
                    'text': text,
                    'publication_date': 'Unknown',  # Refine if date extraction needed
                    'url': url
                })
                seen_urls.add(url)
                logging.info(f"Collected: {title} - {url}")
            time.sleep(1)  # Rate limiting
    
    articles_data = all_articles[:50]  # Cap at 50 total articles
    logging.info(f"Collected {len(articles_data)} relevant articles for {query}")
    return articles_data
