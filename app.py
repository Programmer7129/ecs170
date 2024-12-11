# from flask import Flask, request, render_template
# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# import pandas as pd
# from preprocess import preprocess_image

# # Initialize Flask app
# app = Flask(__name__)
# UPLOAD_FOLDER = 'static/images'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load the trained model
# model = load_model('model/dog_breed_model.h5')

# # Load the breed info dataset
# breed_info = pd.read_csv('dogs-ranking-dataset.csv')

# # Automatically generate class indices by parsing folder names
# image_folder = 'dataset/dataset/Images'
# class_indices = {
#     folder.split('-')[-1]: idx
#     for idx, folder in enumerate(sorted(os.listdir(image_folder)))
# }
# reverse_class_indices = {v: k for k, v in class_indices.items()}

# def preprocess_breed_name(class_name):
#     class_name = class_name.replace('_', ' ')
#     if class_name.endswith(' dog'):
#         class_name = class_name[:-4]
#     return class_name.title()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return render_template('index.html', error="No file uploaded")

#     file = request.files['file']
#     if file.filename == '':
#         return render_template('index.html', error="No file selected")

#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(file_path)

#     img_array = preprocess_image(file_path)
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions[0])

#     breed_name = preprocess_breed_name(reverse_class_indices.get(predicted_class, "Unknown"))

    # breed_data = breed_info[breed_info['Breed'].str.casefold() == breed_name.casefold()]
    # if not breed_data.empty:
    #     breed_details = breed_data.to_dict(orient='records')[0]
    # else:
    #     breed_details = {"Breed": breed_name, "Info": "No additional details available."}

    # return render_template(
    #     'result.html',
    #     image_url=file.filename,
    #     breed=breed_name,
    #     confidence=round(float(predictions[0][predicted_class]) * 100, 2),
    #     breed_details=breed_details
    # )

# if __name__ == '__main__':
#     app.run(debug=True)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import requests
from io import BytesIO
from preprocess import preprocess_image  # Your custom preprocess logic
import json

# Initialize Flask app
app = Flask(__name__)

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
    """Clean up the breed name for display"""
    class_name = class_name.replace('_', ' ')
    if class_name.endswith(' dog'):
        class_name = class_name[:-4]
    return class_name.title()

@app.route('/')
def index():
    """ Renders the main page with the upload form """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """ 
    Receives the Cloudinary image URL, predicts the breed, 
    and returns the result to be displayed on the results page 
    """
    data = request.get_json()  # Get JSON data from the POST request
    image_url = data.get('image_url')  # Extract image URL from JSON request
    
    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400

    # 🔥 Download the image from Cloudinary
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        image_bytes = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            image_bytes.write(chunk)
    except Exception as e:
        return jsonify({'error': f'Error fetching image from URL: {str(e)}'}), 400

    # 🔥 Preprocess the image using your existing preprocess logic
    img_array = preprocess_image(image_bytes)

    # 🔥 Use the model to predict the breed
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  # Get the index of the highest predicted probability
    confidence_score = round(float(predictions[0][predicted_class]) * 100, 2)  # Get the confidence score

    # 🔥 Convert class index to breed name
    breed_name = preprocess_breed_name(reverse_class_indices.get(predicted_class, "Unknown"))

    # 🔥 Get breed details from the dataset
    breed_data = breed_info[breed_info['Breed'].str.casefold() == breed_name.casefold()]
    if not breed_data.empty:
        breed_details = breed_data.to_dict(orient='records')[0]
    else:
        breed_details = {"Breed": breed_name, "Info": "No additional details available."}

    # session['breed_details'] = breed_details

    return jsonify({
        'image_url': image_url,
        'breed': breed_name,
        'confidence': confidence_score,
        'breed_details': breed_details
    })

@app.route('/results')
def results():
    """ 
    Renders the results page, displaying the uploaded image and breed prediction 
    """
    image_url = request.args.get('image_url')
    breed = request.args.get('breed')
    confidence = request.args.get('confidence')

    # Ensure breed_details is passed to the template
    breed_details = request.args.get('breed_details', '{}')  # Default to an empty JSON string
    # breed_details = session.pop('breed_details', {})
    if breed_details:
        try:
            breed_details = json.loads(breed_details)  # Convert JSON string back to Python dictionary
        except Exception as e:
            print(f"Error parsing breed_details: {e}")
            breed_details = {}  # If parsing fails, set to empty dictionary

    if not image_url or not breed or not confidence:
        return "Missing image, breed, or confidence information.", 400

    return render_template(
        'result.html',
        image_url=image_url,
        breed=breed,
        confidence=confidence,
        breed_details=breed_details  # Pass breed_details as a dictionary
    )


# Required to run the app
if __name__ == '__main__':
    app.run(debug=True)
