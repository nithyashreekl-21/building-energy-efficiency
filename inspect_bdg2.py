import urllib.request
import ssl
import json

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
headers = {'User-Agent': 'Mozilla/5.0'}

url = 'https://api.github.com/repos/buds-lab/building-data-genome-project-2'
req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
        data = json.loads(r.read().decode('utf-8'))
        print("BDG2 Repo Description:", data.get('description'))
except Exception as e:
    print("Error:", e)

url_readme = 'https://raw.githubusercontent.com/buds-lab/building-data-genome-project-2/master/README.md'
req_readme = urllib.request.Request(url_readme, headers=headers)
try:
    with urllib.request.urlopen(req_readme, context=ctx, timeout=10) as r:
        content = r.read().decode('utf-8')
        print("\n=== BDG2 README EXCERPT ===")
        print(content[:1500])
except Exception as e:
    print("Readme Error:", e)
