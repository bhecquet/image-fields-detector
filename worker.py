'''
Created on 26 oct. 2022

@author: S047432
'''

import base64
import logging
import os
import sys
import tempfile

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.results.backends.redis import RedisBackend
from dramatiq.results.middleware import Results
from unidecode import unidecode

from TextProcessor import TextProcessor
from config import *
from logging.config import dictConfig


# from flask_melodramatiq import RedisBroker
processors = {'field_processor': None,
              'error_processor': None}

def configure():
    
    os.makedirs('logs', exist_ok=True)
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }},
        'handlers': {
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'default',
                'filename': 'logs/detector.log',
                'backupCount': 3,
                'maxBytes': 4000
                }
            },
        'root': {
            'level': 'INFO',
            'handlers': ['file']
        }
    })
    
    # adding current directory in path because 'flask worker' puts only parent directory in path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from CleaningThread import CleaningThread
    from Processor import Processor
    
    if unidecode(os.path.abspath(IMAGE_PATH)) != os.path.abspath(IMAGE_PATH):
        raise Exception("Image Path must not contain accents")
        
        
    logging.info("loading field detector model from " + FIELD_DETECTOR_WEIGHTS)
    fd_processor = Processor(FIELD_DETECTOR_WEIGHTS,
                            INFO_PATH,
                            DEVICE,
                            fake_mode=FAKE,
                            archive_mode=ARCHIVE)
                              
    logging.info("loading error detector model from " + ERROR_DETECTOR_WEIGHTS)
    ed_processor = Processor(ERROR_DETECTOR_WEIGHTS,
                            INFO_PATH,
                            DEVICE,
                            fake_mode=FAKE,
                            archive_mode=ARCHIVE)
                              
    CleaningThread(INFO_PATH).start()
        
        
    UPLOAD_FOLDER = os.path.abspath(IMAGE_PATH) + os.sep + 'api'
    OUTPUT_FOLDER = os.path.abspath(INFO_PATH) 
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    return fd_processor, ed_processor

broker = RedisBroker(url=DRAMATIQ_BROKER_URL)
dramatiq.set_broker(broker)
result_backend = RedisBackend(url=DRAMATIQ_BROKER_URL)
broker.add_middleware(Results(backend=result_backend))

fd_processor, ed_processor = configure()
processors['field_processor'] = fd_processor
processors['error_processor'] = ed_processor

@dramatiq.actor(store_results=True, max_age=10000)
def detect_remote(processor_name, imageb64, image_name, resize_factor):
    """
    @param processor_name: name of the processor to use
    @param imageb64: image transmitted as a Base 64 string
    @param image_name: name of the file
    @param resize_factor: factor to apply to picture, in case it's small (for example)
    """
    # image file is transmitted as base 64 string
    with open(os.path.join(tempfile.gettempdir(), image_name), 'wb') as image:
        bytestring = base64.b64decode(imageb64.encode('utf-8'))
        image.write(bytestring)

    try:
        return processors[processor_name].detect(image.name, resize_factor)
    except Exception as e:
        os.unlink(image.name)
        
    return {'error': None}


@dramatiq.actor(store_results=True, max_age=10000)
def detect_text_remote(imageb64, image_name):
    """
    Detect text on picture
    @param imageb64: image transmitted as a Base 64 string
    @param image_name: name of the file
    """
    # image file is transmitted as base 64 string
    with open(os.path.join(tempfile.gettempdir(), image_name), 'wb') as image:
        bytestring = base64.b64decode(imageb64.encode('utf-8'))
        image.write(bytestring)

    try:
        text_processor = TextProcessor('eng')
        return text_processor.get_text_boxes(image.name)
    except Exception as e:
        os.unlink(image.name)
        
    return {'error': None}

# Start with either
# - python -m dramatiq worker:broker --processes 1
# - dramatiq worker:broker --processes 1
# - python dramatiq_worker.py worker:broker --processes 1 (for usage in IDE)

