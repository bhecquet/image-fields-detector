'''
Created on 26 oct. 2022

@author: S047432
'''

from app import create_app, broker

app = create_app()

# Start with either
# - python -m dramatiq worker:broker --processes 1
# - dramatiq worker:broker --processes 1
# - python dramatiq_worker.py worker:broker --processes 1 (for usage in IDE)

