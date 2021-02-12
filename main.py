import argparse
import logging
from _ssl import txt2obj
from numpy import source
from image_processor.Processor import Processor
import time
import torch
import os


def delete_old_files(path):
    
    for f in os.listdir(path):
        
        if time.time() - os.stat(os.path.join(path, f)).st_mtime > 60 * 60 * 24:
            try:
                os.remove(os.path.join(path, f))
            except Exception as e:
                logging.warn("cannot delete file: " + f)
    


if __name__ == '__main__':

    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Tool for reading JUnit results and generating results')
    parser.add_argument('img_path', help="Path where pictures should be placed")
    parser.add_argument('info_path', help="Path where information about detected zones will be published")
    parser.add_argument('--weights', type=str, default='best_web-generated-yolov3-spp_50.pt', help='weights path')
    parser.add_argument('--cfg', type=str, default='web-generated-yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='web-generated.names', help='*.names path')
    parser.add_argument('--archive', default=False, type=bool, help='whether to archive picture and discovery in archve folder')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--fake', default=False, type=bool, help='whether to simulate zone discovery')
    opt = parser.parse_args()

    processor = Processor(opt.weights,
                          opt.cfg,
                          opt.names,
                          opt.info_path,
                          opt.device,
                          image_size=1600,
                          fake_mode=opt.fake,
                          archive_mode=opt.archive)
    
    
    while True:
        
        with torch.no_grad():
            processor.detect(opt.img_path)
        
        # wait
        time.sleep(5)
        
        # delete old processed files
        delete_old_files(opt.info_path)
