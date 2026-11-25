import json
import numpy as np
from pathlib import Path

import tensorflow as tf
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, url_for

BASE_DIR = Path(__file__).resolve.parent
MODEL_PATH = BASE_DIR / 'model' / 'rps.keras'
CLASSES_PATH = BASE_DIR / 'model' / 'class_names.json'

UPLOAD_FOLDER = BASE_DIR / 'static' 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] =  str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGHT'] = 10 * 1024 * 1024  #10 MB

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, 'r', encoding='utf8') as f:
    classes_names = json.load(f)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1).lower() in ALLOWED_EXTENSIONS


def predict_image(filepath: Path) -> tuple[str, float]:
    img = tf.keras.utils.load_img(filepath, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)
    idx = int(np.argmax(predictions[0]))
    confidence = round(float(np.max(predictions[0])) * 100, 2 )

    return classes_names[idx], confidence


@app.route('/', methods=['GET', 'POST'])
def index_view():
    if request.method == 'POST':
        file = request.files.get('file')
        if not allowed_file(file.filename):
            return render_template('index.html', error="Not allowed format")

        filename = secure_filename(file.filename)
        save_path = UPLOAD_FOLDER / filename
        file.save(save_path)

        prediction, confidence = predict_image(save_path)
        image_url = url_for('static', filename=f'uploads/{filename}')

        return render_template('index.html', image_url=image_url, prediction=prediction, confidence=confidence)
    else:
        return render_template('index.html')
