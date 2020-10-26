import requests
import cv2

url = 'http://127.0.0.1:5000/predict'
my_img = {'image': cv2.imread('test.jpg')}
r = requests.post(url, files=my_img)

# convert server response into JSON format.
print(r.json()) 

