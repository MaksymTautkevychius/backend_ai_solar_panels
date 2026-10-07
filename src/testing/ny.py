import requests

ACCOUNT_URI = "https://3ca55f76-efd8-4df4-bbb8-6434293ffedc.northeurope.account.maps.azure.com"
CLIENT_ID   = "3ca55f76-efd8-4df4-bbb8-6434293ffedc"
TOKEN       = "YOUR_AAD_BEARER_TOKEN"

url = f"{ACCOUNT_URI}/map/static/png"

params = {
    "api-version": "2024-04-01",
    "center": "21.0122,52.2297",   # lon,lat
    "zoom": 14,
    "width": 800,
    "height": 600,
    "layer": "basic",
    "style": "main"
}

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "x-ms-client-id": CLIENT_ID
}

response = requests.get(url, params=params, headers=headers)

with open("map.png", "wb") as f:
    f.write(response.content)

print("Saved map.png")