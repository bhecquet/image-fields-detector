'''
Created on 7 juin 2021

@author: S047432
'''
from flask import Flask, request, send_from_directory
import argparse
import logging
import os
import time
import torch

from Processor import Processor
from werkzeug.utils import secure_filename
import threading
from werkzeug.exceptions import abort
from unidecode import unidecode
from flask.helpers import make_response
from PIL import Image
from logging.config import dictConfig


dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {
        'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
            },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'default',
            'filename': 'detector.log',
            'backupCount': 3,
            'maxBytes': 4000
            }
        },
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi', 'file']
    }
})

app = Flask(__name__)
fd_processor = None # processor for field detection
ed_processor = None # processor for error detection

                
class CleaningThread(threading.Thread):
    
    def __init__(self, output_directory):
        self.output_directory = output_directory
        super(CleaningThread, self).__init__()
        
    def delete_old_files(self, path):
    
        for f in os.listdir(path):
            
            file_path = os.path.join(path, f)
            if time.time() - os.stat(file_path).st_mtime > 60 * 60 * 24 and os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logging.warning("cannot delete file: " + f)
    
    def run(self):
        while True:

            # wait
            time.sleep(5)
            
            # delete old processed files
            self.delete_old_files(self.output_directory)
            
def detect(processor):
    """
    Method to send an image and get detection data (POST)
    
    to test: curl -F "image=@D:\Dev\yolo\yolov3\dataset_generated_small\out-7.jpg"   http://127.0.0.1:5000/detect
    """
    
    if request.method == 'POST':
        
        
        if 'image' not in request.files:
            logging.error("image not found")
            abort(400, description="image not found")
        factor = request.form.get('resize', 1, type=float)

        file = request.files['image']
        logging.info("detecting fields for image %s" % file)
        
        if file.filename == '':
            abort(400, description="File name is empty")
        
        filename = secure_filename(file.filename)
        
        saved_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(saved_file)

        detection_data = processor.detect(saved_file, factor)
        if not detection_data:
            return make_response({'error': "Error in detection"}, 500)
        elif detection_data['error']:
            return make_response(detection_data, 500)
        else:
            return detection_data
      

    
@app.route('/status', methods = ['GET'])
def status():
    return "OK"

@app.route('/detect', methods = ['POST'])
def detect_fields():
    """
    Method to send an image and get detection data (POST)
    
    to test: curl -F "image=@D:\Dev\yolo\yolov3\dataset_generated_small\out-7.jpg"   http://127.0.0.1:5000/detect
    """
    
    return detect(fd_processor)

@app.route('/detectError', methods = ['POST'])
def detect_error():
    """
    Method to send an image and get detection data for errors in forms (POST)
    
    to test: curl -F "image=@D:\Dev\yolo\yolov3\dataset_generated_small\out-7.jpg"   http://127.0.0.1:5000/detectError
    """
    
    return detect(ed_processor)

    
@app.route('/debug', methods = ['GET', 'POST'])
def get_debug_file():
    """
    to test: curl http://127.0.0.1:5000/debug?image=out-7.jpg --output toto.jpg
    """
    
    if request.method == 'GET':
        
        image_name = request.args.get('image')
        
        if image_name is None:
            abort(400, description="'image' parameter is mandatory")
        elif image_name in os.listdir(app.config['OUTPUT_FOLDER']):
            return send_from_directory(app.config["OUTPUT_FOLDER"], image_name)
        elif unidecode.unidecode(image_name.replace(' ', '_')) in os.listdir(app.config['OUTPUT_FOLDER']):
            return send_from_directory(app.config["OUTPUT_FOLDER"], unidecode.unidecode(image_name.replace(' ', '_')))
        else:
            abort(404, description="image '%s' not found" % image_name)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Tool for reading JUnit results and generating results')
    parser.add_argument('img_path', help="Path where pictures should be placed")
    parser.add_argument('info_path', help="Path where information about detected zones will be published")
    parser.add_argument('--fd-weights', type=str, default='weights/field_detector.pt', help='weights path for field detection')
    parser.add_argument('--ed-weights', type=str, default='weights/error_detector.pt', help='weights path for error detection')
    parser.add_argument('--archive', default=False, type=bool, help='whether to archive picture and discovery in archve folder')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--fake', default=False, type=bool, help='whether to simulate zone discovery')
    parser.add_argument('--language', default='eng', type=str, help='detection language')
    opt = parser.parse_args()
    
    if unidecode(os.path.abspath(opt.img_path)) != os.path.abspath(opt.img_path):
        raise Exception("Image Path must not contain accents")
    
    
    logging.info("loading field detector model from " + opt.fd_weights)
    fd_processor = Processor(opt.fd_weights,
                          opt.info_path,
                          opt.device,
                          fake_mode=opt.fake,
                          archive_mode=opt.archive)
    
    logging.info("loading error detector model from " + opt.ed_weights)
    ed_processor = Processor(opt.ed_weights,
                          opt.info_path,
                          opt.device,
                          fake_mode=opt.fake,
                          archive_mode=opt.archive)
    
    CleaningThread(opt.info_path).start()

    
    app.config['UPLOAD_FOLDER'] = os.path.abspath(opt.img_path) + os.sep + 'api'
    app.config['OUTPUT_FOLDER'] = os.path.abspath(opt.info_path) 
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    app.run(host="0.0.0.0")
