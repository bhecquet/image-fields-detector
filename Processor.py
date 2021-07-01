'''
Created on 22 oct. 2020

@author: S047432
'''

import logging

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from xml.dom import minidom
import collections
import shutil
import random
from pathlib import Path
import os
import time
import torch
import traceback
import cv2
from utils.plots import plot_one_box
from TextProcessor import TextProcessor
import json


ObjectBox = collections.namedtuple('ObjectBox', ['class_id', 'x_min', 'x_max', 'y_min', 'y_max'])

class ObjectBox:
    def __init__(self, class_id, class_name, left, right, top, bottom):
        self.class_id = int(class_id)
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.class_name = class_name
        self.text = None
        self.related_field = None
        self.with_label = class_name.endswith('with_label')
        self.width = right - left
        self.height = bottom - top
        
    def to_dict(self):
        d = vars(self)
        if self.related_field is not None:
            d['related_field'] = vars(self.related_field)
        return d
        
    def __eq__(self, other):
        return self.top == other.top and self.left == other.left and self.width == other.width and self.height == other.height 

        

class Processor:
    
    def __init__(self, 
                 weights,
                 output_directory,
                 device='', 
                 batch_size=16, 
                 max_image_size=1664,  
                 iou_thres=0.45,  # for nms
                 conf_thres=0.25,
                 fake_mode=False,
                 archive_mode=False,
                 language='eng'
                 ):
        """
        @param weights: 'pt' file representing the computed weights during training
        @param chg: cfg file, configuration of the model
        @param names: the ".names" file defining classes
        @param output_directory: directory where to write detection information and resulting images
        @param device: device id (i.e. 0 or 0,1) or cpu
        @param batch_size: size of each image batch
        @param max_image_size: max image size for inference
        @param iou_thres: IOU threshold for NMS
        @param conf_thres: object confidence threshold
        @param fake_mode: if True, will simulate the class detection but will not perform it. Model will not be initialized
        @param archive_mode: if True, write the detected picture and classes to the archive folder
        @param language: 'eng', 'fra', ... The language tesseract will use to find text
        """
        self.weights = weights
        self.fake_mode = fake_mode
        self.device = device
        self.batch_size = batch_size
        self.max_image_size = max_image_size
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.output_directory = output_directory

        self.verbose = True
        self.augment = False
        self.agnostic_nms = False
        self.archive_directory = None
  
        if not self.fake_mode:
            self.device, self.model, self.colors = self._init_model()
            
        if archive_mode:
            self.archive_directory = Path(output_directory).parent.joinpath('archives')
            self.archive_directory.mkdir(exist_ok=True, parents=True)
            
        self.language = language
    
    def _init_model(self):
        """
        Load model
        
        @return: the device that will do computing
        """
        
        device = select_device(self.device)

        # Initialize model
        model = attempt_load(self.weights, map_location=device)  # load FP32 model

        self.half = device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            model.half()  # to FP16
    
        # Get names and colors
        self.names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
            
        return device, model, colors
    
    @torch.no_grad()
    def detect(self, source):
        """
        Detect area in JPG image
        
        @param source: the image source (folder or single file)
        """
        
        detection_data = {'error': None}

        try:
            dataset = LoadImages(source, 1600, stride = int(self.model.stride.max())) 
        except AssertionError as e:
            detection_data['error'] = str(e)
            return detection_data
        
        if self.fake_mode:
            for path, img, im0s, vid_cap in dataset:
                save_path = str(Path(self.output_directory) / Path(path).name)
                class_file = save_path[:save_path.rfind('.')] + '.txt'
                with open(class_file, 'a') as file:
                    xywh = [0.352140, 0.275000, 0.380026, 0.086842]
                    cls = 0
                    file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    
                if self.archive_directory:
                    shutil.copy(class_file, self.archive_directory)
                    shutil.copy(path, self.archive_directory)
                    self.create_xml_class_file(self.archive_directory, Path(path).name, img.shape[0], img.shape[1], boxes)
                    
                # remove image file
                os.remove(path)
                    
                    
        else:

            t0 = time.time()
            
            for path, img, im0s, vid_cap in dataset:
                
                # set image size on model, for each image because detection is better is model image size almost matches the image size
                image_size = dataset.img_size 
                
                if self.device.type != 'cpu':
                    self.model(torch.zeros(1, 3, image_size, image_size).to(self.device).type_as(next(self.model.parameters())))  # run once
   
                try:
                    
                    detection_data_for_img = self.detect_fields(path, img, im0s)
                    detection_data[Path(path).name] = detection_data_for_img
                
                except Exception as e:
                    traceback.print_exc()
                    logging.error("could not apply detection on {}".format(path))
                    
                # remove image file
                os.remove(path)
        
            print('Results saved to %s' % self.output_directory)

            print('Done. (%.3fs)' % (time.time() - t0))
            
        return detection_data
    
    def detect_fields(self, path, img, im0s):
        """
        Detect fields on image
        @param path: path to the image
        @param img: image on which we search fields
        @param im0s: image with detected fields
        """
        save_path = str(Path(self.output_directory) / Path(path).name)
        class_file = save_path[:save_path.rfind('.')] + '.txt'
        boxes = []
        save_img = True
        
        try:
            os.remove(class_file)
        except:
            pass
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, agnostic=self.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            p, s, im0 = path, '', im0s
            
            img0_width = im0.shape[0]
            img0_height = im0.shape[1]

            s += '%gx%g ' % img.shape[2:] 
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                
                # Rescale boxes from self.image_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    
                    # write boxes to text file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    with open(class_file, 'a') as file:
                        file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                        
                    xmin, ymin, xmax, ymax = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                    boxes.append(ObjectBox(cls.int(), self.names[int(cls)], int(xmin), int(xmax), int(ymin), int(ymax)))

                    if save_img:  # Add bbox to image
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)
                
        # create a file that can be used by labelImg to reinject this picture in training
        self.create_xml_class_file(self.output_directory, Path(path).name, img0_width, img0_height, boxes)
           
        # archive image and detected classes     
        if self.archive_directory:
            shutil.copy(class_file, self.archive_directory)
            shutil.copy(path, self.archive_directory)
            self.create_xml_class_file(self.archive_directory, Path(path).name, img0_width, img0_height, boxes)
            
        # match fields with dependency relation
        self.correlate_fields_with_labeled_fields(boxes)
            
        # process text on image to match detected boxes with labels
        text_processor = TextProcessor(self.language)
        text_boxes = text_processor.get_text_boxes(path)
        self.correlate_text_and_fields(text_boxes, boxes)
            
        # create output file
        detection_data_for_img = {'fields': [b.to_dict() for b in boxes], 'labels': [vars(b) for b in text_boxes.values()]}
        with open(os.path.join(self.output_directory, Path(path).stem + '.json'), 'w') as json_file:
            json_file.write(json.dumps(detection_data_for_img))
            
        return detection_data_for_img

    def correlate_fields_with_labeled_fields(self, field_boxes):
        """
        As Yolo detection will produce fields "with_label" and simple fields (ex: checkbox with and without label are 2 fields), we try to 
        match both 
        """
        for field_box_with_label in [f for f in field_boxes if f.with_label]:
            for field_box_no_label in [f for f in field_boxes if not f.with_label]:
                x_box_center = (field_box_no_label.right - field_box_no_label.left) / 2 + field_box_no_label.left
                y_box_center = (field_box_no_label.bottom - field_box_no_label.top) / 2 + field_box_no_label.top
                
                if (x_box_center > field_box_with_label.left
                    and x_box_center < field_box_with_label.right
                    and y_box_center > field_box_with_label.top
                    and y_box_center < field_box_with_label.bottom):
                    field_box_with_label.related_field = field_box_no_label
                    break

    def correlate_text_and_fields(self, text_boxes, field_boxes):
        """
        Try to match a box discovered by tesseract and a box discovered by field recognition
        """
        
        for name, text_box in text_boxes.items():
            
            x_text_box_center = (text_box.right - text_box.left) / 2 + text_box.left
            y_text_box_center = (text_box.bottom - text_box.top) / 2 + text_box.top
            
            for field_box in field_boxes:
                
                if (x_text_box_center > field_box.left
                    and x_text_box_center < field_box.right
                    and y_text_box_center > field_box.top
                    and y_text_box_center < field_box.bottom
                    and self.names[int(field_box.class_id)].endswith("with_label")):
                    field_box.text = text_box.text
                    break


    def create_xml_class_file(self, folder, filename, image_width, image_height, boxes):
        
        
        form = minidom.parseString("""
        <annotation>
        <folder>archive</folder>
        <filename>{}</filename>
        <path>{}/{}</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>{}</width>
            <height>{}</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        </annotation>
        """.format(filename, folder, filename, image_width, image_height))
        
        for box in boxes:
            object = minidom.parseString("""<object>
            <name>{}</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>{}</xmin>
                <ymin>{}</ymin>
                <xmax>{}</xmax>
                <ymax>{}</ymax>
            </bndbox>
        </object>""".format(self.names[box.class_id], box.left, box.top, box.right, box.bottom))
            form.childNodes[0].appendChild(object.childNodes[0])
            
        xml_content = form.toxml()
            
        os.path.splitext(filename)[0]
        with open(os.path.join(folder, os.path.splitext(filename)[0] + '.xml'), 'w') as xml_file:
            xml_file.write(xml_content)