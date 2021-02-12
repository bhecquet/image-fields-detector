'''
Created on 22 oct. 2020

@author: S047432
'''

import argparse
import json
import logging

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *
from xml.dom import minidom
import collections


ObjectBox = collections.namedtuple('ObjectBox', ['class_id', 'x_min', 'x_max', 'y_min', 'y_max'])

class Processor:
    
    def __init__(self, 
                 weights, 
                 cfg, 
                 names, 
                 output_directory,
                 device='', 
                 batch_size=16, 
                 image_size=512,  
                 iou_thres=0.6,  # for nms
                 conf_thres=0.3,
                 fake_mode=False,
                 archive_mode=False
                 ):
        """
        @param weights: 'pt' file representing the computed weights during training
        @param chg: cfg file, configuration of the model
        @param names: the ".names" file defining classes
        @param output_directory: directory where to write detection information and resulting images
        @param device: device id (i.e. 0 or 0,1) or cpu
        @param batch_size: size of each image batch
        @param image_size: inference size (pixels). The larger are the images, the higher should be this value, for accuracy
        @param iou_thres: IOU threshold for NMS
        @param conf_thres: object confidence threshold
        @param fake_mode: if True, will simulate the class detection but will not perform it. Model will not be initialized
        @param archive_mode: if True, write the detected picture and classes to the archive folder
        """
        self.weights = weights
        self.cfg = cfg
        self.fake_mode = fake_mode
        self.device = device
        self.batch_size = batch_size
        self.image_size = image_size
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.output_directory = output_directory
        
        self.names_file = names
        
        self.verbose = True
        self.half = True  # half precision FP16 inference
        self.augment = False
        self.agnostic_nms = False
        self.archive_directory = None
        self.names = load_classes(self.names_file)
  
        if not self.fake_mode:
            self.device, self.model, self.colors = self._init_model()
            
        if archive_mode:
            self.archive_directory = Path(output_directory).parent.joinpath('archives')
            self.archive_directory.mkdir(exist_ok=True, parents=True)
    
    def _init_model(self):
        """
        Load model
        
        @return: the device that will do computing
        """
        
        device = torch_utils.select_device(self.device, batch_size=self.batch_size)
        
        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # Initialize model
        model = Darknet(self.cfg, self.image_size)

        # Load weights
        if self.weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(self.weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, self.weights)
            
        # Eval mode
        model.to(self.device).eval()
    
        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()
    
        # Half precision
        self.half = self.half and device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            model.half()
    
    
        # Get names and colors
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
            
        return device, model, colors
    
    def detect(self, source):
        """
        Detect area in JPG image
        
        @param source: the image source
        """
        
        # Set Dataloader
        save_img = True
        boxes = []
        
        try:
            dataset = LoadImages(source, img_size=self.image_size)
        except AssertionError as e:
            return
        
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
        # TODO: init de img pourrait être déplacée dans l'init du modèle
            # Run inference
            t0 = time.time()
            img = torch.zeros((1, 3, self.image_size, self.image_size), device=self.device)  # init img
            _ = self.model(img.half() if self.half else img.float()) if self.device.type != 'cpu' else None  # run once
            
            for path, img, im0s, vid_cap in dataset:
                
                boxes = []
                try:
                    
                    save_path = str(Path(self.output_directory) / Path(path).name)
                    class_file = save_path[:save_path.rfind('.')] + '.txt'
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
                    t1 = torch_utils.time_synchronized()
                    pred = self.model(img, augment=self.augment)[0]
                    t2 = torch_utils.time_synchronized()
            
                    # to float
                    if self.half:
                        pred = pred.float()
            
                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                               multi_label=False, classes=[], agnostic=self.agnostic_nms)
            
                    # Process detections
                    for i, det in enumerate(pred):  # detections for image i
                        p, s, im0 = path, '', im0s
                        
                        img_width = img.shape[2]
                        img_height = img.shape[3]
                        img0_width = im0.shape[0]
                        img0_height = im0.shape[1]
            
                        s += '%gx%g ' % (img_width, img_height)  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                        if det is not None and len(det):
                            
                            # Rescale boxes from self.image_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            
                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += '%g %ss, ' % (n, self.names[int(c)])  # add to string
            
                            # Write results
                            for *xyxy, conf, cls in det:
                                
                                # write boxes to text file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                with open(class_file, 'a') as file:
                                    file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                                    
                                xmin, ymin, xmax, ymax = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                                boxes.append(ObjectBox(cls.int(), int(xmin), int(xmax), int(ymin), int(ymax)))
            
                                if save_img:  # Add bbox to image
                                    label = '%s %.2f' % (self.names[int(cls)], conf)
                                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)])
            
                        # Print time (inference + NMS)
                        print('%sDone. (%.3fs)' % (s, t2 - t1))
            
                        # Save results (image with detections)
                        if save_img:
                            cv2.imwrite(save_path, im0)
                            
                    self.create_xml_class_file(self.output_directory, Path(path).name, img0_width, img0_height, boxes)
                       
                    # archive image and detected classes     
                    if self.archive_directory:
                        shutil.copy(class_file, self.archive_directory)
                        shutil.copy(path, self.archive_directory)
                        self.create_xml_class_file(self.archive_directory, Path(path).name, img0_width, img0_height, boxes)
                        
                    # remove image file
                    os.remove(path)
                
                except Exception as e:
                    logging.error("could not apply detection on {}".format(path))
        
            print('Results saved to %s' % self.output_directory)
            if platform == 'darwin':  # MacOS
                os.system('open ' + save_path)
        
            print('Done. (%.3fs)' % (time.time() - t0))


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
        </object>""".format(self.names[box.class_id], box.x_min, box.y_min, box.x_max, box.y_max))
            form.childNodes[0].appendChild(object.childNodes[0])
            
        xml_content = form.toxml()
            
        with open(os.path.join(folder, filename.replace('.jpg', '.xml')), 'w') as xml_file:
            xml_file.write(xml_content)