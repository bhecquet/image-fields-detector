'''
Created on 27 oct. 2022

@author: 
'''

from flask_melodramatiq import create_broker_class
from dramatiq.brokers.redis import RedisBroker
from flask_melodramatiq.lazy_broker import LAZY_BROKER_DOCSTRING_TEMPLATE
import threading
import psutil
import logging
import time
import statistics
from config import CPU_LIMIT

class ResourceAwareRedisBroker(RedisBroker):
    
    lock = threading.Lock()
    too_busy = 0
    
    def __init__(self, *args, **kwds):
        self.resource_monitor = ResourceMonitorThread()
        self.resource_monitor.start()
        super().__init__(*args, **kwds)
    
    def do_fetch(self, *args, **kwds):

        load = self.resource_monitor.get_cpu_load()
        
        with ResourceAwareRedisBroker.lock:
        
            if load > CPU_LIMIT and ResourceAwareRedisBroker.too_busy < 5 :
                logging.info(f"too busy {load}: {ResourceAwareRedisBroker.too_busy}")
                ResourceAwareRedisBroker.too_busy += 1
                time.sleep(0.1) # we are busy, let other worker take tasks
                return []
            else:
                ResourceAwareRedisBroker.too_busy = 0
                return self._dispatch('fetch')(*args, **kwds)
        
   
class ResourceMonitorThread(threading.Thread):

    lock = threading.Lock()

    def __init__(self, *args, **kwds):
        self.loads = []
        self.cpu_load = psutil.cpu_percent(interval=1)
        self.loads.append(self.cpu_load)
        super().__init__(*args, **kwds)

    def run(self):
        while True:
            
            load = psutil.cpu_percent(interval=1)
            self.loads = self.loads[-59:]
            self.loads.append(load)
            
            with ResourceMonitorThread.lock:
                self.cpu_load = statistics.mean(self.loads)
            
    def get_cpu_load(self):
        with ResourceMonitorThread.lock:
            return self.cpu_load
            
        
# RedisBroker = create_broker_class(
    # classpath='ResourceAwareRedisBroker:ResourceAwareRedisBroker',
    # docstring=LAZY_BROKER_DOCSTRING_TEMPLATE.format(
        # description='A lazy broker wrapping a :class:`~dramatiq.brokers.redis.RedisBroker`.\n',
    # ),
# )
