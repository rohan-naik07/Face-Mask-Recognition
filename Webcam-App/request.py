from cv2 import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask,redirect,jsonify,request
from PIL import Image
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)

path_model=os.path.join(os.getcwd(),'res10_300x300_ssd_iter_140000.caffemodel')
path_protxt=os.path.join(os.getcwd(),'deploy.prototxt.txt')

net = cv2.dnn.readNetFromCaffe(path_protxt ,path_model)
MODEL_PATH =os.path.join(os.getcwd(),'model3.h5')
MODEL_PATH2 =os.path.join(os.getcwd(),'model3.json')

json_file=open(MODEL_PATH2,'r')
loaded_model_json=json_file.read()
json_file.close()

model=tf.keras.models.model_from_json(loaded_model_json)
model.load_weights(MODEL_PATH)

conf = 0.70
no_conf = 1.99

@app.route('/predict',methods=['POST'])
def predict():
    img_file = request.files['image']
    image = Image.open(img_file.stream) 
    blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)), 1.0 ,(300,300) ,(104.0, 177.0, 123.0))
    (h,w) = image.shape[:2]
    net.setInput(blob)
    detections = net.forward()

    for i2 in range(0 , detections.shape[2]):
        confidence = detections[0,0,i2,2]

        if (confidence > conf) and (confidence < no_conf) :
            box = detections[0,0,i2,3:7]*np.array([w,h,w,h])
            (startX , startY , endX , endY) = box.astype("int")
            (startX, startY) = (max(0, startX),max(0, startY))
            (endX, endY) = min(w - 1, endX), min(h - 1, endY)

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face)[0]

            label="No Mask"
            color=(0,0,255)

            if mask < withoutMask:
                label="Mask" 
                color=(0,255,0)  
		

    context = {
    'label' : label,
    'color' : color,
    'start_coor' : [int(startX),int(startY)],
    'end_coor' : [int(endX),int(endY)] 
    }

    return jsonify(context)

if __name__ == "__main__":
    app.run(debug=True)





