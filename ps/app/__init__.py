import os
import sys
from flask import Flask

app = Flask(__name__)

""" Get absolute path to resource, works for dev and for PyInstaller """
if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
else:
    application_path = os.getcwd()

import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_directory_path = os.path.join(os.getcwd(), 'logs')
log_file_path = os.path.join(log_directory_path, 'PyServicesApp.log')

if not os.path.exists(log_directory_path):
    os.makedirs(log_directory_path)

file_handler = RotatingFileHandler(log_file_path, maxBytes=10485760, backupCount=10)
file_handler.setFormatter(Formatter(
    '%(asctime)s %(levelname)s: %(message)s '
    '[in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

from app import routes
#from app import calclib
from app import calculation