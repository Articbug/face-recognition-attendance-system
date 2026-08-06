import requests

url = "http://127.0.0.1:5000/recognize"

image_path = r"C:\Users\chand\Documents\Projects\Face-Recognize-Model\3. Models\backend\test.jpeg" # put any image here

files = {
    "image": open(image_path, "rb")
}

response = requests.post(url, files=files)

print(response.json())