#!/usr/bin/env python3
import logging
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

logger = logging.getLogger('hexhex')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# handler for file output
if config.get('LOGGING', 'file', fallback="") != "":
    fh = logging.FileHandler(config.get('LOGGING', 'file'), mode=config.get('LOGGING', 'file_mode'))
    fh.setLevel(config.get('LOGGING', 'file_level'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# handle for console output
ch = logging.StreamHandler()
ch.setLevel(config.get('LOGGING', 'console_level', fallback="INFO"))

# create formatter and add it to the handlers
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)
logger.propagate = False
