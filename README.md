# image-fields-detector
This project aims at finding text fields, combobox, button, etc in an image and associate it to its label
It provides 2 models:
- weights/field_detector.pt: finds elements on an image (text field, button, ...)
- weights/error_detector.pt: finds error messages


## Installation

image-field-detector is a worker that aims to be used with SeleniumRobot-server [https://github.com/bhecquet/seleniumRobot-server](https://github.com/bhecquet/seleniumRobot-server)

### Redis

Install Redis

- edit redis.conf and comment the "bind 127.0.0.1" to listen to all interface (if it's what you need) => also set protection mode to "no" in case you are in a internal network

### Dependencies

Clone the git repository and then (you should use a virtualenv)

`pip install -r requirements`

## Configuration

### SeleniumRobot-server

Configure Dramatiq broker with

```
DRAMATIQ_BROKER['OPTIONS']['url'] = 'redis://<redis_server>:6379/0'
DRAMATIQ_RESULT_BACKEND['BACKEND_OPTIONS']['url'] = 'redis://<redis_server>:6379'
```
in settings.py file

### Image-field-detector

Edit 'config.py' file to set the redis URL

```
DRAMATIQ_BROKER_URL = 'redis://<redis_server>:6379/0'
CPU_LIMIT = 10
```

## Usage

Start with either
- python -m dramatiq worker:broker --processes 1
- dramatiq worker:broker --processes 1
- python dramatiq_worker.py worker:broker --processes 1 (for usage in IDE)

Logs are written in 'logs' folder

Then, use

```
curl -F "image=@<path_to_your_image.jpg" -F "task=field"   http://<selenium_robot_server>:<port>/snapshot/detect/
```