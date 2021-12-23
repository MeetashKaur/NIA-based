import os
import sys
# Flask
#import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
from os import listdir
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import matplotlib.pyplot as plt
# Some utilites
import numpy as np
from util import base64_to_pil
import pickle
from xgboost import XGBClassifier
from tensorflow.keras.applications.vgg19 import VGG19
#from googletrans import Translator, constants
#from pprint import pprint
# Declare a flask app
app = Flask(__name__)



model= pickle.load(open('recog_model.pkl','rb'))
print('Model loaded. Start serving...')



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        VGG19_model=VGG19(weights='imagenet',include_top=False,input_shape=(224,224,3))
        for layer in VGG19_model.layers:
        	layer.trainable = False 
			
        img = base64_to_pil(request.json)
        # fig=cv2.imread(os.path.join("E:/test_images",'9.jpg'))
        # n_image= cv2.resize(fig,(224,224))
        n_image= img.resize((224, 224))
        f_image = np.reshape(n_image,[1,224,224,3])
        print(f_image)
        input_img_extractor=VGG19_model.predict(f_image)
        input_img_features=input_img_extractor.reshape(input_img_extractor.shape[0], -1)
        input_pred=model.predict(input_img_features)
	  
        numbers_list = {0: "0", 1: "੧", 2: "੨", 3: "੩" , 4: "੪", 5 : "੫",
        6: "੬", 7: "੭", 8: "੮" , 9: "੯", 10 : "੧0"}

        PredValue = int(input_pred)
        result = numbers_list[PredValue]
        # translator = Translator()
        # result = translator.translate(str(result), dest='pa')
        return jsonify(result=result)
    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
   
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
