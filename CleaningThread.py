'''
Created on 26 oct. 2022

@author: 
'''
import threading
import os
import time
import logging

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
