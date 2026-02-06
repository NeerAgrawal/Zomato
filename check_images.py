import requests
import re

# The mapping from web_app.py
CUISINE_IMAGES = {
    'north indian': [
        'https://images.unsplash.com/photo-1585937421612-70a008356f36?w=800&q=80',
        # ... (I will include only a subset to test or read the file directly)
    ]
}
# Actually I will read web_app.py to get the real dict
import ast

def extract_urls(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple regex to find urls
    urls = re.findall(r'https://images.unsplash.com/[^\s\'"]+', content)
    return list(set(urls)) # Dedupe

def check_urls():
    urls = extract_urls('phase5/web_app.py')
    print(f"checking {len(urls)} images...")
    
    valid = []
    invalid = []
    
    for url in urls:
        try:
            r = requests.head(url, timeout=2)
            if r.status_code == 200:
                print(f"OK: {url}")
                valid.append(url)
            else:
                print(f"FAIL ({r.status_code}): {url}")
                invalid.append(url)
        except Exception as e:
            print(f"ERR: {url} - {e}")
            invalid.append(url)
            
    print(f"\nSummary: {len(valid)} Valid, {len(invalid)} Invalid")

if __name__ == "__main__":
    check_urls()
