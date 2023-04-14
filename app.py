'''
Created on 7 juin 2021

@author: S047432
'''
import logging
from logging.config import dictConfig
import os

from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend
from flask import Flask, request, send_from_directory
from flask.helpers import make_response
from unidecode import unidecode
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename
import sys
import dramatiq
# from flask_melodramatiq import RedisBroker
from ResourceAwareRedisBroker import RedisBroker


def configure(app):
    # adding current directory in path because 'flask worker' puts only parent directory in path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from CleaningThread import CleaningThread
    from Processor import Processor
    
    if unidecode(os.path.abspath(app.config['IMAGE_PATH'])) != os.path.abspath(app.config['IMAGE_PATH']):
        raise Exception("Image Path must not contain accents")
        
        
    logging.info("loading field detector model from " + app.config['FIELD_DETECTOR_WEIGHTS'])
    fd_processor = Processor(app.config['FIELD_DETECTOR_WEIGHTS'],
                            app.config['INFO_PATH'],
                            app.config['DEVICE'],
                            fake_mode=app.config['FAKE'],
                            archive_mode=app.config['ARCHIVE'])
                              
    logging.info("loading error detector model from " + app.config['ERROR_DETECTOR_WEIGHTS'])
    ed_processor = Processor(app.config['ERROR_DETECTOR_WEIGHTS'],
                            app.config['INFO_PATH'],
                            app.config['DEVICE'],
                            fake_mode=app.config['FAKE'],
                            archive_mode=app.config['ARCHIVE'])
                              
    CleaningThread(app.config['INFO_PATH']).start()
        
        
    app.config['UPLOAD_FOLDER'] = os.path.abspath(app.config['IMAGE_PATH']) + os.sep + 'api'
    app.config['OUTPUT_FOLDER'] = os.path.abspath(app.config['INFO_PATH']) 
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    return fd_processor, ed_processor

broker = RedisBroker()
dramatiq.set_broker(broker)

@dramatiq.actor(store_results=True)
def say_hello(url):
    print(f"Hello {url!r}.")
    return "done"
      
def create_app(init_processors=True):
    
    app = Flask(__name__)
    app.config.from_pyfile('config.py')
    broker.init_app(app)
    result_backend = RedisBackend(host="zld7205v")
    broker.add_middleware(Results(backend=result_backend))
    if init_processors:
        fd_processor, ed_processor = None, None#configure(app)
    
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
    
    @app.route('/testRedis', methods = ['GET'])
    def test_redis():
        message = say_hello.send("bla")
        return message.get_result(block=True, backend=result_backend)
        
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
            
    return app
            
if __name__ == '__main__':
    
    # configure logging
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
    
    app = create_app(init_processors=False)
    app.run(host="0.0.0.0")
    #broker = dramatiq.broker