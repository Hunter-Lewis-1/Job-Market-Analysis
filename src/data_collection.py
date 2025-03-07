from gnews import GNews
from newspaper import Article as NewspaperArticle
import trafilatura
import requests
from bs4 import BeautifulSoup
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def collect_news_articles(query, use_dummy=False):
    """
    Collects news articles using GNews, fetching up to 100 articles per year from 2021 to 2025.
    For 'Traba', ensures articles are relevant to the staffing company using specific keywords.
    
    Args:
        query (str): Company name to search (e.g., 'Traba').
        use_dummy (bool): If True, returns dummy data; if False, collects real data.
    
    Returns:
        list: List of dicts with keys: 'title', 'text', 'publication_date', 'url', 'keywords', 'summary'.
    """
    articles_data = []
    
    if use_dummy:
        logging.info(f"Using dummy data for {query}")
        return [
            {'title': f"{query} Raises Funds", 'text': f"{query} secures $20M for staffing innovation.",
             'publication_date': '2025-03-01', 'url': 'http://example.com/dummy1',
             'keywords': [query.lower(), 'funding'], 'summary': f"{query} gets funding."},
            {'title': f"{query} Expands", 'text': f"{query} grows operations in NY.",
             'publication_date': '2025-03-02', 'url': 'http://example.com/dummy2',
             'keywords': [query.lower(), 'expansion'], 'summary': f"{query} expands."}
        ]
    
    years = [2021, 2022, 2023, 2024, 2025]
    all_articles = []
    
    for year in years:
        if year < 2025:
            end_date = (year, 12, 31)
        else:
            end_date = None
        try:
            logging.info(f"Fetching articles for {query} from {year}")
            gnews = GNews(language='en', country='US', start_date=(year, 1, 1), end_date=end_date, max_results=100)
            articles = gnews.get_news(query)
            all_articles.extend(articles)
            logging.info(f"Fetched {len(articles)} articles for {year}")
            time.sleep(5)
        except Exception as e:
            logging.error(f"Error fetching articles for {year}: {e}")
            continue
    
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        url = article.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)
    
    logging.info(f"Total unique articles fetched: {len(unique_articles)}")
    
    # Custom headers to mimic a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/'
    }
    
    for i, article in enumerate(unique_articles):
        rss_url = article.get('url', '')
        actual_url = article.get('publisher', {}).get('href', rss_url)
        
        if 'news.google.com' in actual_url.lower():
            logging.info(f"Resolving URL for article {i}: {rss_url}")
            try:
                response = requests.get(rss_url, timeout=10, headers=headers)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    canonical = soup.find('link', rel='canonical')
                    if canonical and canonical.get('href'):
                        actual_url = canonical['href']
                    else:
                        refresh = soup.find('meta', attrs={'http-equiv': 'refresh'})
                        if refresh and refresh.get('content'):
                            url_part = refresh['content'].split('url=')[-1]
                            if url_part:
                                actual_url = url_part
                logging.info(f"Resolved URL: {actual_url}")
            except Exception as e:
                logging.error(f"Failed to resolve URL for article {i} at {rss_url}: {e}")
                continue
        
        title = article.get('title', '')
        publication_date = article.get('published date', '')
        
        text = ''
        keywords = []
        summary = ''
        # Newspaper3k with custom headers
        for attempt in range(3):
            try:
                logging.info(f"Extracting text with Newspaper3k for article {i}: {actual_url}")
                news_article = NewspaperArticle(actual_url, headers=headers)
                news_article.download()
                news_article.parse()
                text = news_article.text if news_article.text else ''
                if text:
                    news_article.nlp()
                    keywords = news_article.keywords if news_article.keywords else []
                    summary = news_article.summary if news_article.summary else ''
                break
            except Exception as e:
                logging.error(f"Newspaper3k failed for article {i} at {actual_url}, attempt {attempt + 1}: {e}")
                time.sleep(2)
        
        # Trafilatura with custom headers
        if not text:
            for attempt in range(3):
                try:
                    logging.info(f"Extracting text with trafilatura for article {i}: {actual_url}")
                    downloaded = trafilatura.fetch_url(actual_url, headers=headers)
                    if downloaded:
                        text = trafilatura.extract(downloaded) or ''
                    else:
                        text = ''
                    keywords = []
                    summary = ''
                    break
                except Exception as e:
                    logging.error(f"Trafilatura failed for article {i} at {actual_url}, attempt {attempt + 1}: {e}")
                    time.sleep(2)
        
        if not text:
            logging.warning(f"Article {i} skipped: No text extracted for {actual_url} (Title: {title})")
            continue
        
        if query.lower() not in title.lower() and query.lower() not in text.lower():
            logging.warning(f"Article {i} skipped: '{query}' not in title or text")
            continue
        
        if query.lower() == 'traba':
            relevance_keywords = [
                'staffing', 'workforce', 'employment', 'gig economy',
                'temporary workers', 'on-demand labor'
            ]
            if not any(keyword in text.lower() or keyword in title.lower() for keyword in relevance_keywords):
                logging.warning(f"Article {i} skipped for {query}: No relevance keywords found")
                continue
        
        articles_data.append({
            'title': title,
            'text': text,
            'publication_date': publication_date,
            'url': actual_url,
            'keywords': keywords,
            'summary': summary
        })
        logging.info(f"Article {i} collected: Title='{title}', Text length={len(text)}, URL={actual_url}")
    
    logging.info(f"Collected {len(articles_data)} relevant articles for {query} since 2021")
    return articles_data
