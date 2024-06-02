from flask import Flask 
from flask import request 
from flask import jsonify
import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as k 
from keras.models import Sequential 
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
classes=["apple_scab","apple_black_rot","apple_black_rot","apple_healthy","blueberry_healthy","cherry_healthy","cherry_powdery_mildew","corn_cercospora_leaf_spot","corn_common_rust","corn_healthy","corn_northern_leaf_blight","grape_black_rot","grape_black_measles","grape_healthy","grape_leaf_blight","orange_haunglongbing","beach_bacterial_spot","peach_healthy","pepper-bell_bacterial_spot","pepper-bell_healthy","potato_early_blight","potato_healthy","potato_late_blight","rasberry_healthy","soybean_healthy","squach_powdery_mildew","starwberry_healthy","strawberry_leaf_scorch","tomato_bacterial_spot","tomato_early_blight","tomato_healthy","tomato_late_blight","tomato_leaf_mold","tomato_septoria_leaf_spot","tomato_spider_mites","tomato_target_spot","tomato_mosaic_virus","tomato_yellow_leaf_curl_virus"]
app=Flask(__name__)
def get_model():
    global model 
    model=load_model('model.keras')
    print("* model loaded ")
def preprocess_image(image,target_size):
    if image.mode !="RGB":
        image=image.convert("RGB")
    image=image.resize(target_size)
    image= img_to_array(image)
    image=np.expand_dims(image,axis=0)
    return image 

print("loading keras model")
get_model()    
@app.route('/predict',methods=['POST'])
def prdict():
    message=request.get_json(force=True)
    
    encoded=message['image']
    decoded=base64.b64decode(encoded)
    image=Image.open(io.BytesIO(decoded))
    processed_image=preprocess_image(image,target_size=(128,128))
    prediction=model.predict(processed_image).tolist()
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = classes[predicted_class_index]
    
    print(f"Prediction: {predicted_class_name} with confidence {prediction[0][predicted_class_index]}")
    
    response = {
        'prediction': predicted_class_name
    }
    return jsonify(response)



