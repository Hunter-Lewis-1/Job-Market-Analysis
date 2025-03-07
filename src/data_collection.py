from gnews import GNews
from newspaper import Article as NewspaperArticle
import trafilatura
import requests
from bs4 import BeautifulSoup
import time

def collect_news_articles(query, use_dummy=False):
    """
    Collects news articles using GNews for discovery and Newspaper3k or trafilatura for content extraction.
    For 'Traba', ensures articles are relevant to the staffing company using specific keywords.
    
    Args:
        query (str): Company name to search (e.g., 'Traba').
        use_dummy (bool): If True, returns dummy data; if False, collects real data.
    
    Returns:
        list: List of dicts with keys: 'title', 'text', 'publication_date', 'url', 'keywords', 'summary'.
    """
    articles_data = []
    
    # Dummy data for testing
    if use_dummy:
        print(f"Using dummy data for {query}")
        return [
            {
                'title': f"{query} Raises Funds",
                'text': f"{query} secures $20M for staffing innovation.",
                'publication_date': '2025-03-01',
                'url': 'http://example.com/dummy1',
                'keywords': [query.lower(), 'funding'],
                'summary': f"{query} gets funding."
            },
            {
                'title': f"{query} Expands",
                'text': f"{query} grows operations in NY.",
                'publication_date': '2025-03-02',
                'url': 'http://example.com/dummy2',
                'keywords': [query.lower(), 'expansion'],
                'summary': f"{query} expands."
            }
        ]
    
    # Define years to query
    years = [2021, 2022, 2023, 2024, 2025]
    all_articles = []
    
    for year in years:
        if year < 2025:
            end_date = (year, 12, 31)
        else:
            end_date = None  # For current year, fetch up to present
        try:
            gnews = GNews(language='en', country='US', start_date=(year, 1, 1), end_date=end_date, max_results=100)
            articles = gnews.get_news(query)
            all_articles.extend(articles)
            print(f"Fetched {len(articles)} articles for {query} from {year}")
            time.sleep(2)  # Delay to avoid rate limits
        except Exception as e:
            print(f"Error fetching articles for {query} from {year}: {e}")
    
    # Remove duplicates based on URL
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        url = article.get('url', '')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)
    
    print(f"Total unique articles fetched for {query}: {len(unique_articles)}")
    
    # Process each article
    for i, article in enumerate(unique_articles):
        # Get the URL from the article - try to get the real URL
        rss_url = article.get('url', '')
        actual_url = article.get('publisher', {}).get('href', rss_url)
        
        # If still using RSS URL, try to extract the real URL
        if 'news.google.com' in actual_url:
            try:
                response = requests.get(rss_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                soup = BeautifulSoup(response.text, 'html.parser')
                # Look for the canonical link or a redirect
                canonical = soup.find('link', rel='canonical')
                if canonical and canonical.get('href'):
                    actual_url = canonical['href']
                else:
                    # Look for meta refresh tag as alternative
                    refresh = soup.find('meta', attrs={'http-equiv': 'refresh'})
                    if refresh and refresh.get('content'):
                        content = refresh['content']
                        if 'url=' in content.lower():
                            actual_url = content.split('url=')[-1].strip()
            except Exception as e:
                print(f"Failed to resolve real URL for article {i} at {rss_url}: {e}")
        
        title = article.get('title', '')
        publication_date = article.get('published date', '')
        
        # Extract content with Newspaper3k with retry logic
        text = ''
        keywords = []
        summary = ''
        
        # Retry logic for Newspaper3k extraction
        for attempt in range(3):
            try:
                news_article = NewspaperArticle(actual_url)
                news_article.download()
                news_article.parse()
                text = news_article.text if news_article.text else ''
                if text:
                    news_article.nlp()
                    keywords = news_article.keywords if news_article.keywords else []
                    summary = news_article.summary if news_article.summary else ''
                    break  # Success, exit retry loop
                else:
                    print(f"Empty text from Newspaper3k for article {i}, attempt {attempt + 1}")
                    time.sleep(2)  # Wait before retry
            except Exception as e:
                print(f"Newspaper3k failed for article {i} at {actual_url}, attempt {attempt + 1}: {e}")
                time.sleep(2)  # Wait before retry
        
        # Fallback to trafilatura if Newspaper3k fails with retry logic
        if not text:
            for attempt in range(3):
                try:
                    downloaded = trafilatura.fetch_url(actual_url)
                    if downloaded:
                        text = trafilatura.extract(downloaded) or ''
                        if text:
                            break  # Success, exit retry loop
                        else:
                            print(f"Empty text from trafilatura for article {i}, attempt {attempt + 1}")
                            time.sleep(2)  # Wait before retry
                    else:
                        print(f"Failed to download with trafilatura for article {i}, attempt {attempt + 1}")
                        time.sleep(2)  # Wait before retry
                except Exception as e:
                    print(f"Trafilatura failed for article {i} at {actual_url}, attempt {attempt + 1}: {e}")
                    time.sleep(2)  # Wait before retry
        
        # Skip if no text extracted
        if not text:
            print(f"Article {i} skipped: No text extracted for {actual_url}")
            continue
        
        # Relevance check: Ensure company name is in title or text
        if query.lower() not in title.lower() and query.lower() not in text.lower():
            print(f"Article {i} skipped: '{query}' not in title or text")
            continue
        
        # Traba-specific relevance check
        if query.lower() == 'traba':
            relevance_keywords = [
                'staffing', 'workforce', 'employment', 'gig economy',
                'temporary workers', 'on-demand labor'
            ]
            if not any(keyword in text.lower() or keyword in title.lower() for keyword in relevance_keywords):
                print(f"Article {i} skipped for {query}: No relevance keywords found")
                continue
        
        # Add article to results
        articles_data.append({
            'title': title,
            'text': text,
            'publication_date': publication_date,
            'url': actual_url,  # Store the resolved URL
            'keywords': keywords,
            'summary': summary
        })
        print(f"Article {i} collected: Title='{title}', Text length={len(text)}, Date={publication_date}")
    
    print(f"Collected {len(articles_data)} relevant articles for {query}")
    return articles_data
