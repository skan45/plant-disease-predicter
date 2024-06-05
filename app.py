from flask import Flask, request, jsonify
import base64
import numpy as np
import io
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from flask_cors import CORS
import threading
import time
# Define the classes
classes = [
    "apple_scab", "apple_black_rot", "apple_rust", "apple_healthy", "blueberry_healthy",
    "cherry_healthy", "cherry_powdery_mildew", "corn_cercospora_leaf_spot",
    "corn_common_rust", "corn_healthy", "corn_northern_leaf_blight", "grape_black_rot",
    "grape_black_measles", "grape_healthy", "grape_leaf_blight", "orange_haunglongbing",
    "peach_bacterial_spot", "peach_healthy", "pepper-bell_bacterial_spot", "pepper-bell_healthy",
    "potato_early_blight", "potato_healthy", "potato_late_blight", "raspberry_healthy",
    "soybean_healthy", "squash_powdery_mildew", "strawberry_healthy", "strawberry_leaf_scorch",
    "tomato_bacterial_spot", "tomato_early_blight", "tomato_healthy", "tomato_late_blight",
    "tomato_leaf_mold", "tomato_septoria_leaf_spot", "tomato_spider_mites", "tomato_target_spot",
    "tomato_mosaic_virus", "tomato_yellow_leaf_curl_virus"
]

# Initialize the Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3001"}, r"/latest_prediction": {"origins": "http://localhost:3001"}})

# Function to load the model
def get_model():
    global model
    model = load_model('model.keras')
    print("* Model loaded")

# Function to preprocess the image
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

print("Loading Keras model")
get_model()

# Global variable to store the latest prediction
latest_prediction = ""
def clear_prediction():
    global latest_prediction
    time.sleep(60)  # Sleep for one minute
    latest_prediction = ""

# Start the thread to clear the prediction
thread = threading.Thread(target=clear_prediction)
thread.start()

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    global latest_prediction
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(128, 128))
    prediction = model.predict(processed_image).tolist()
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = classes[predicted_class_index]
    
    # Store the latest prediction
    latest_prediction = predicted_class_name

    print(f"Prediction: {predicted_class_name} with confidence {prediction[0][predicted_class_index]}")
    
    response = {
        "prediction": predicted_class_name
    }
    print(response)
    return jsonify(response)

# Define the route to get the latest prediction
@app.route('/latest_prediction', methods=['GET'])
def get_latest_prediction():
    response = {
        "prediction": latest_prediction
    }
    return jsonify(response)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
