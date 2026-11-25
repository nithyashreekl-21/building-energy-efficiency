import urllib.request
import ssl
import re

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
headers = {'User-Agent': 'Mozilla/5.0'}

req = urllib.request.Request('https://data.mendeley.com/datasets/m6vv9k9gcd/5', headers=headers)
with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
    html = r.read().decode('utf-8', errors='ignore')
    
    # Description
    desc_match = re.search(r'Description(.*?)Steps to reproduce', html, re.DOTALL)
    if desc_match:
        text = re.sub(r'<[^>]+>', ' ', desc_match.group(1))
        text = ' '.join(text.split())
        print("=== MENDELEY DESCRIPTION ===")
        print(text[:1500])
        
    # Steps to reproduce
    steps_match = re.search(r'Steps to reproduce(.*?)Institutions', html, re.DOTALL)
    if steps_match:
        text = re.sub(r'<[^>]+>', ' ', steps_match.group(1))
        text = ' '.join(text.split())
        print("\n=== STEPS TO REPRODUCE ===")
        print(text[:1000])

    # Find file names
    print("\n=== FILES FOUND ===")
    file_matches = re.findall(r'href="([^"]*files[^"]*)"[^>]*>([^<]+)</a>', html)
    for link, name in file_matches[:15]:
        print(f"File: {name.strip()} -> {link}")
