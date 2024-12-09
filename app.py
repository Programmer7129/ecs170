from flask import Flask, request, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from preprocess import preprocess_image

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('model/dog_breed_model.h5')

# Load the breed info dataset
breed_info = pd.read_csv('dogs-ranking-dataset.csv')

# Automatically generate class indices by parsing folder names
image_folder = 'dataset/dataset/Images'
class_indices = {
    folder.split('-')[-1]: idx
    for idx, folder in enumerate(sorted(os.listdir(image_folder)))
}
reverse_class_indices = {v: k for k, v in class_indices.items()}

def preprocess_breed_name(class_name):
    class_name = class_name.replace('_', ' ')
    if class_name.endswith(' dog'):
        class_name = class_name[:-4]
    return class_name.title()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    breed_name = preprocess_breed_name(reverse_class_indices.get(predicted_class, "Unknown"))

    breed_data = breed_info[breed_info['Breed'].str.casefold() == breed_name.casefold()]
    if not breed_data.empty:
        breed_details = breed_data.to_dict(orient='records')[0]
    else:
        breed_details = {"Breed": breed_name, "Info": "No additional details available."}

    return render_template(
        'result.html',
        image_url=file.filename,
        breed=breed_name,
        confidence=round(float(predictions[0][predicted_class]) * 100, 2),
        breed_details=breed_details
    )

