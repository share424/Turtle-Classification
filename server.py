import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from preprocess import remove_background, get_glcm_feature, get_his_feature
from PIL import Image
import cv2

UPLOAD_FOLDER = "templates/static/uploads/"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app._static_folder = os.path.abspath("templates/static/")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/klasifikasi2')
def klasifikasi():
    return render_template('klasifikasi.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/klasifikasi', methods=['POST'])
def preprocess():
    if 'file' not in request.files:
        return request.files
    
    file = request.files['file']
    filename = "image.jpg"
    clean_filename = "clean.jpg"
    glcm_filename = "glcm.jpg"
    his_filename = "his.jpg"
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # preprocess
    image = Image.open(UPLOAD_FOLDER+"/"+filename)
    image = np.array(image)

    
    clean_image = remove_background(image)

    glcm_features = get_glcm_feature(clean_image)
    his_features = get_his_feature(clean_image)
    
    image_features = []
    for glcm in glcm_features:
        image_features.append(glcm)
    for his in his_features:
        image_features.append(his)

    # save fake image
    glcm_image = cv2.cvtColor(clean_image, cv2.COLOR_RGB2GRAY)
    glcm_image = Image.fromarray(glcm_image)
    his_image = cv2.cvtColor(clean_image, cv2.COLOR_RGB2HLS)
    his_image = Image.fromarray(his_image)
    clean_image = Image.fromarray(clean_image)

    clean_image.save(UPLOAD_FOLDER+clean_filename)
    glcm_image.save(UPLOAD_FOLDER+glcm_filename)
    his_image.save(UPLOAD_FOLDER+his_filename)

    res = {"image_features": image_features}
    # return res
    return render_template('klasifikasi.html', image_features=image_features)

if __name__ == '__main__':
    app.run(debug=True)