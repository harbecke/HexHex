#!/usr/bin/env python3
import logging
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

logger = logging.getLogger('hexhex')
logger.setLevel(logging.DEBUG)

# handler for file output
fh = logging.FileHandler(config.get('LOGGING', 'file'))
fh.setLevel(config.get('LOGGING', 'file_level'))

# handle for console output
ch = logging.StreamHandler()
ch.setLevel(config.get('LOGGING', 'console_level'))

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
