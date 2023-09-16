import requests

content = {"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}

result = requests.post("http://127.0.0.1:5000/api/flower", json=content)

print(result.status_code)
print(result.text)