import requests
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

alpha_key = config["data"]["api_keys"]["alpha_vantage"]

url = 'https://www.alphavantage.co/query'
params = {
    'function': 'TIME_SERIES_INTRADAY',
    'symbol': 'GOOGL',
    'interval': '5min',
    'apikey': alpha_key,
    'outputsize': 'full',
    'datatype': 'json'
}

r = requests.get(url, params=params)
print(r.url)
print(r.status_code)
print(r.json())
