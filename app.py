import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import pandas as pd
import requests
from io import BytesIO
from preprocess import preprocess_image

# Initialize Flask app
app = Flask(__name__)

# Configure TensorFlow for memory efficiency
physical_devices = tf.config.list_physical_devices('CPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Lazy loading variables
model = None
breed_info = None
class_indices = None
reverse_class_indices = None

def load_model_and_data():
    """Lazy load model and data only when needed"""
    global model, breed_info, class_indices, reverse_class_indices
    
    if model is None:
        # Load model with memory-efficient settings
        model = tf.keras.models.load_model(
            'model/dog_breed_model.h5',
            compile=False  # Don't compile model on load
        )
        
        # Load breed info
        breed_info = pd.read_csv('dogs-ranking-dataset.csv')
        
        # Generate class indices
        image_folder = 'dataset/dataset/Images'
        class_indices = {
            folder.split('-')[-1]: idx
            for idx, folder in enumerate(sorted(os.listdir(image_folder)))
        }
        reverse_class_indices = {v: k for k, v in class_indices.items()}

def preprocess_breed_name(class_name):
    """Clean up the breed name for display"""
    class_name = class_name.replace('_', ' ')
    if class_name.endswith(' dog'):
        class_name = class_name[:-4]
    return class_name.title()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load model and data only when needed
        load_model_and_data()
        
        data = request.get_json()
        image_url = data.get('image_url')
        
        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400

        # Download and process image
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image_bytes = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            image_bytes.write(chunk)

        # Preprocess image
        img_array = preprocess_image(image_bytes)

        # Make prediction
        with tf.device('/CPU:0'):  # Force CPU usage
            predictions = model.predict(img_array, batch_size=1)
        
        predicted_class = np.argmax(predictions[0])
        confidence_score = round(float(predictions[0][predicted_class]) * 100, 2)
        
        # Get breed name and details
        breed_name = preprocess_breed_name(reverse_class_indices.get(predicted_class, "Unknown"))
        breed_data = breed_info[breed_info['Breed'].str.casefold() == breed_name.casefold()]
        
        breed_details = breed_data.to_dict(orient='records')[0] if not breed_data.empty else {
            "Breed": breed_name,
            "Info": "No additional details available."
        }

        return jsonify({
            'image_url': image_url,
            'breed': breed_name,
            'confidence': confidence_score,
            'breed_details': breed_details
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results')
def results():
    image_url = request.args.get('image_url')
    breed = request.args.get('breed')
    confidence = request.args.get('confidence')
    breed_details = request.args.get('breed_details', '{}')

    try:
        import json
        breed_details = json.loads(breed_details)
    except:
        breed_details = {}

    if not all([image_url, breed, confidence]):
        return "Missing required information.", 400

    return render_template(
        'result.html',
        image_url=image_url,
        breed=breed,
        confidence=confidence,
        breed_details=breed_details
    )

if __name__ == '__main__':
    app.run(debug=True)
