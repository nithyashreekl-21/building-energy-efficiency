import urllib.request
import re
import ssl
import json

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def inspect_url(name, url):
    print(f"\n==========================================")
    print(f"INSPECTING: {name}")
    print(f"URL: {url}")
    print(f"==========================================")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
            html = r.read().decode('utf-8', errors='ignore')
            title = re.search(r'<title>(.*?)</title>', html, re.I)
            print("Title:", title.group(1).strip() if title else "N/A")
            
            # JSON-LD
            json_lds = re.findall(r'<script type=["\']application/ld\+json["\']>(.*?)</script>', html, re.DOTALL)
            for j in json_lds:
                try:
                    parsed = json.loads(j)
                    if isinstance(parsed, dict) and 'name' in parsed:
                        print("JSON-LD Name:", parsed.get('name'))
                        print("JSON-LD Description:", str(parsed.get('description'))[:300])
                except Exception:
                    pass
            
            # Clean text sample
            text = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL)
            text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = ' '.join(text.split())
            print("Summary snippet:", text[:500])
    except Exception as e:
        print("Error fetching:", e)

# 1. Mendeley
inspect_url("Urban Building Energy Stock Dataset", "https://data.mendeley.com/datasets/m6vv9k9gcd/5")

# 2. Building Data Genome Project 2
inspect_url("Building Data Genome Project 2", "https://ngci.encs.concordia.ca/ckan/dataset/building-data-genome-project-2")

# 3. ASHRAE Great Energy Predictor III
inspect_url("ASHRAE Great Energy Predictor III", "https://www.kaggle.com/competitions/ashrae-energy-prediction/data")
