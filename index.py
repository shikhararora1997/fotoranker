import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from photofeature import perform
from scipy.stats import entropy
import cv2
import numpy as np
import pandas as pd
import os
import glob
import math
from PIL import Image
from PIL import ImageStat



app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads/'
# ALLOWED_EXTENSIONS =  set(['jpg', 'png', 'csv'])
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
@app.route("/", methods=['GET','POST'])

def home():
    return render_template('layout.html')



@app.route("/upload", methods=['GET','POST'])
def upload(): 
    target = os.path.join(APP_ROOT, 'images/')
    

    if not os.path.isdir(target):
        os.mkdir(target)
    filenames=[]
    for file in request.files.getlist("file"):
        
        filename = file.filename
        
        destination = "".join([target, filename])
        print(destination)
        file.save(destination)
        filenames.append(destination)
  
    perform(filenames)
    
    return render_template("complete.html")
            
    # file=request.files['file']
    # filename=file.filename
    # return filename
    # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


if __name__ == '__main__':
    app.run()