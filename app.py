'''
Created on 7 juin 2021

@author: S047432
'''
from flask import Flask, request, flash, redirect
import argparse
import logging
import os
import time
import torch

from Processor import Processor
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)
processor = None

                
class ProcessorThread(threading.Thread):
    
    def __init__(self, processor, input_directory, output_directory):
        self.processor = processor
        self.output_directory = output_directory
        self.input_directory = input_directory
        super(ProcessorThread, self).__init__()
        
    def delete_old_files(self, path):
    
        for f in os.listdir(path):
            
            if time.time() - os.stat(os.path.join(path, f)).st_mtime > 60 * 60 * 24:
                try:
                    os.remove(os.path.join(path, f))
                except Exception as e:
                    logging.warn("cannot delete file: " + f)
    
    def run(self):
        while True:

            self.processor.detect(self.input_directory)
                
            # wait
            time.sleep(5)
            
            # delete old processed files
            self.delete_old_files(self.output_directory)
    

@app.route('/detect', methods = ['POST'])
def upload_file():
    """
    to test: curl -F "image=@D:\Dev\yolo\yolov3\dataset_generated_small\out-7.jpg"   http://127.0.0.1:5000/detect
    """
    
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['image']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        
        saved_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(saved_file)
        return processor.detect(saved_file)

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Tool for reading JUnit results and generating results')
    parser.add_argument('img_path', help="Path where pictures should be placed")
    parser.add_argument('info_path', help="Path where information about detected zones will be published")
    parser.add_argument('--weights', type=str, default='best_web-generated-yolov3-spp_50.pt', help='weights path')
    parser.add_argument('--archive', default=False, type=bool, help='whether to archive picture and discovery in archve folder')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--fake', default=False, type=bool, help='whether to simulate zone discovery')
    parser.add_argument('--language', default='eng', type=str, help='detection language')
    opt = parser.parse_args()
    
    
    processor = Processor(opt.weights,
                          opt.info_path,
                          opt.device,
                          fake_mode=opt.fake,
                          archive_mode=opt.archive)
    
    ProcessorThread(processor, opt.img_path, opt.img_path).start()

    
    app.config['UPLOAD_FOLDER'] = os.path.abspath(opt.img_path) + os.sep + 'api'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    app.run()