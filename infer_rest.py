import requests
if 1:
    path = {"path":'_datasets/test/test01.png'}
    resp = requests.post("http://localhost:5000/predict", data=path)
else:
    files = {"file": open('drone/test/00068_512_4.jpg','rb')}
    resp = requests.post("http://localhost:5000/predict", files=files)
print(resp.status_code)
print(resp.json())
