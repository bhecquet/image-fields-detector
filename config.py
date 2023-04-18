#from dramatiq.brokers.redis import RedisBroker

TESTING = False
IMAGE_PATH = "input"    # Path where pictures to analyze should be placed
INFO_PATH = "output"    # Path where information about detected zones will be published
FIELD_DETECTOR_WEIGHTS = 'weights/field_detector.pt'    # weights path for field detection
ERROR_DETECTOR_WEIGHTS = 'weights/error_detector.pt'    # weights path for error detection
ARCHIVE = False         # whether to archive picture and discovery in archive folder'
DEVICE = ''             # device id (i.e. 0 or 0,1) or cpu => device to use for detection
FAKE = False            # whether to simulate zone discovery
LANGUAGE = 'eng'        # detection language used by tesseract
#DRAMATIQ_BROKER = RedisBroker
DRAMATIQ_BROKER_URL = "${redis_url}" # url of redis 'redis://<server>:6379/0'
CPU_LIMIT = 20          # max CPU load above which no computation will be done
    

