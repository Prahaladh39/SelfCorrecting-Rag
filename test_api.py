import urllib.request
import json

url = 'http://localhost:5000/api/chat'
data = json.dumps({'message': 'What is Q3 revenue?'}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

try:
    response = urllib.request.urlopen(req)
    print(response.read().decode('utf-8'))
except Exception as e:
    print(e)
