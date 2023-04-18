'''
Created on 11 févr. 2021

@author: S047432
'''

from collections import OrderedDict
import logging
import time

from PIL import Image
import cv2
from matplotlib import pyplot as plt
import pytesseract
from pytesseract.pytesseract import Output

import numpy as np


class TextBox:
    
    def __init__(self, top: int, left: int, width: int, height: int, text: str):
        self.top = top
        self.left = left
        self.width = width
        self.height = height
        self.text = text
        self.right = left + width
        self.bottom = top + height
        
    def zoom(self, ratio):
        """
        Resize this TextBox
        """
        self.left = int(self.left * ratio)
        self.top = int(self.top * ratio)
        self.width = int(self.width * ratio)
        self.height = int(self.height * ratio)
        self.right = self.left + self.width
        self.bottom = self.top + self. height
    
    def __eq__(self, other):
        return self.text == other.text and self.top == other.top and self.left == other.left and self.width == other.width and self.height == other.height 
        
    def __str__(self):
        return '{} => (top={}, left={}, width={}, height={})'.format(self.text, self.top, self.left, self.width, self.height)
    
    def to_dict(self):
        return vars(self)
        

class TextProcessor:


    def __init__(self, language: str):
        self.language = language
        
    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def thresholding(self, image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    def remove_noise(self, image):
        return cv2.medianBlur(image,5)
    
    def enlarge(self, image, ratio):
        return cv2.resize(image, (round(image.shape[1] * ratio), round(image.shape[0] * ratio)), interpolation = cv2.INTER_CUBIC) 
    
    # dilation
    def dilate(self, image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.dilate(image, kernel, iterations = 1)
         
    # erosion
    def erode(self, image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.erode(image, kernel, iterations = 1)
     
    # opening - erosion followed by dilation
    def opening(self, image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
       
    def get_text_boxes(self, image_path: str):
        """
        Build text boxes based on tesseract detection
        It should put in the same box, words which are on the same line and close together 
        """
        texts = OrderedDict()
        
        zoom_ratio = 2
        
        source_image = cv2.imread(image_path)
        
        start = time.time()
        grayscale_image = self.grayscale(source_image)
        enlarged_image = self.enlarge(grayscale_image, zoom_ratio)
        logging.info("image processing: " + str(time.time() - start))
        
        start = time.time()
        detected_boxes = pytesseract.image_to_data(enlarged_image, lang=self.language, output_type=Output.DICT)
        logging.info("tesseract: " + str(time.time() - start))
        
        start = time.time()
        for i in range(len(detected_boxes['level'])):
            
            if int(float(detected_boxes['conf'][i])) < 0 or not detected_boxes['text'][i].strip():
                continue
            
            text = detected_boxes['text'][i]
            top = int(detected_boxes['top'][i])
            left = int(detected_boxes['left'][i])
            width = int(detected_boxes['width'][i])
            height = int(detected_boxes['height'][i])
            
            
            if not texts:
                texts[text] = TextBox(top, left, width, height, text)
            
            else:
                previous_text = texts[list(texts)[-1]]
                
                mean_letter_width = width / len(text)
                
                # same line
                if ((top + height / 2) - (previous_text.top + previous_text.height / 2) < min(height, previous_text.height) # compute mean height of the text box
                    and left - (previous_text.left + previous_text.width) < 2 * mean_letter_width):
                    texts.pop(previous_text.text)
                    texts[previous_text.text + ' ' + text] = TextBox(min(top, previous_text.top), 
                                                                     previous_text.left, 
                                                                     left + width - (previous_text.left), 
                                                                     max(top + height, previous_text.top + previous_text.height) - min(top, previous_text.top),
                                                                     previous_text.text + ' ' + text)
                
                else:
                    texts[text] = TextBox(top, left, width, height, text)
                    
        # adapt sizes and position, with respect to zoom ratio
        for box in texts.values():
            box.zoom(1./zoom_ratio)
            
        # for debug
        # for box in texts.values():
            # cv2.rectangle(source_image, (box.left, box.top), (box.left + box.width, box.top + box.height), (0, 255, 0), 1)
        # cv2.imshow('img', source_image)
        logging.info("output processing: " + str(time.time() - start))
        
        texts2 = OrderedDict()
        for k,v in texts.items():
            texts2[k] = v.to_dict()
        return texts2
 