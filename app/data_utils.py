from cgi import FieldStorage
from flask import Blueprint
from typing import Optional, Tuple
from app.constants import CSV_EXTENSION, TRAIN_SIMULATION_DELAY
import time
import logging

bp = Blueprint('data_utils', __name__)

# Logging configuration
logging.basicConfig(level=logging.INFO)

def read_csv_to_dataframe(file_obj):
    # df = pd.read_csv(file_obj)
    return None #df


def validate_and_get_file(request) -> Tuple[Optional[FieldStorage], Optional[str]]:
    if 'file' not in request.files:
        print("DEBUG: No 'file' part in the request.")
        return None, "No file part in the request."

    f = request.files['file']

    if not f.filename.lower().endswith(CSV_EXTENSION):
        return None, "Invalid file format. Only CSV files are allowed."

    return f, None

def validate_and_get_filePair(request) -> Tuple[Optional[FieldStorage], Optional[FieldStorage], Optional[str]]:
    if 'file1' not in request.files or 'file2' not in request.files:
        return None, None, "Both 'file1' and 'file2' parts are required in the request."

    file1 = request.files['file1']
    file2 = request.files['file2']

    if not all(f.filename.lower().endswith(CSV_EXTENSION) for f in [file1, file2]):
        return None, None, "Invalid file format. Both files must be CSV."

    return file1, file2, None