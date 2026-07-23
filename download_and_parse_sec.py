import requests
from bs4 import BeautifulSoup
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python download_and_parse_sec.py <url>")
    sys.exit(1)

url = sys.argv[1]
headers = {
    # SEC EDGAR requires a User-Agent with contact info to prevent blocking
    "User-Agent": "Research Agent/1.0 (tester@example.com)"
}

print(f"Fetching {url}...")
try:
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    html_content = response.text
    print(f"Successfully fetched HTML. Size: {len(html_content)/1024/1024:.2f} MB")
    
    print("Parsing HTML with BeautifulSoup...")
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Simple extraction of text
    text = soup.get_text(separator='\n', strip=True)
    print(f"Total length of extracted text: {len(text)} characters")
    
    # Print a short preview
    print("\n--- PREVIEW ---")
    print(text[:1000])
    
    # Save the raw HTML to a file for structure-aware chunking later
    filename = url.split('/')[-1]
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\nSaved raw HTML to {filename}")
except requests.exceptions.RequestException as e:
    print(f"Error fetching data: {e}")
