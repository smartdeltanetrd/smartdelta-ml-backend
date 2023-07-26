from cgi import FieldStorage
from flask import Blueprint
from typing import Optional, Tuple
from app.constants import CSV_EXTENSION, TRAIN_SIMULATION_DELAY
import pandas as pd
import time
import logging

bp = Blueprint('data_utils', __name__)

# Logging configuration
logging.basicConfig(level=logging.INFO)

def read_csv_to_dataframe(file_obj):
    df = pd.read_csv(file_obj)
    return df


def validate_and_get_file(request) -> Tuple[Optional[FieldStorage], Optional[str]]:
    if 'file' not in request.files:
        print("DEBUG: No 'file' part in the request.")
        return None, "No file part in the request."

    f = request.files['file']

    if not f.filename.lower().endswith(CSV_EXTENSION):
        return None, "Invalid file format. Only CSV files are allowed."

    return f, None
