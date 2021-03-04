import os
from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory
from werkzeug.utils import secure_filename

from keras.models import Sequential, load_model
import keras, sys
import numpy as np
from PIL import Image


classes = ["red", "blue", "yellow", "green", "purple"]
colors = ["赤い", "青い", "黄色い", "緑の", "紫の"]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def color_name(l, x, default=False):
    index_num = l.index(x) if x in l else default
    return colors[index_num]

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/color', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            model = load_model('./static/color_cnn_aug.h5')

            image = Image.open(filepath)
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)

            result_name = color_name(classes, classes[predicted])

            return render_template('result.html', title='結果ページ', predicted=result_name, percentage=str(
                percentage))

    return render_template('color.html', title='投稿ページ')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
