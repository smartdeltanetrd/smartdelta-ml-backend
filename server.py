from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_json_schema import JsonSchema, JsonValidationError
import pathlib
import datetime
import time  # TODO: remove after function implementations added


UPLOAD_DIR = str(pathlib.Path(__file__).resolve().parent / 'data_files')
ALLOWED_EXTENSIONS = ['.csv']

app = Flask(__name__)
app.config['UPLOAD_DIR'] = UPLOAD_DIR
schema = JsonSchema(app)
CORS(app)


def save_uploaded_file(req, prefix):
    # check if the post request has the file part
    if 'file' not in req.files:
        return False, ("No file part", 400)
    f = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if f.filename == '':
        return False, ("No selected file", 400)
    if pathlib.Path(f.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, ("Not allowed file format", 400)
    date_str = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    f.save(pathlib.Path(
        app.config['UPLOAD_DIR']) / f'{prefix}_{date_str}.csv'
    )
    return True, None


@app.route('/anomaly_train', methods=['POST'])
def anomaly_train():
    res, ret = save_uploaded_file(request, 'anomaly_train')
    if not res:
        return ret
    # TODO: training here, adding simulation delay for a while
    time.sleep(15)
    return {'processed_lines': 0}, 200


@app.route('/anomaly_predict', methods=['POST'])
def anomaly_predict():
    save_uploaded_file(request, 'anomaly_predict')
    res, ret = save_uploaded_file(request, 'anomaly_predict')
    if not res:
        return ret
    # TODO: prediction here, adding simulation delay for a while
    time.sleep(5)
    return {'anomalies': []}, 200


@app.route('/next_hop_train', methods=['POST'])
def next_hop_train():
    res, ret = save_uploaded_file(request, 'next_hop_train')
    if not res:
        return ret
    # TODO: training here, adding simulation delay for a while
    time.sleep(15)
    return {'processed_lines': 0}, 200


@app.route('/next_hop_predict', methods=['POST'])
def next_hop_predict():
    res, ret = save_uploaded_file(request, 'next_hop_predict')
    if not res:
        return ret
    # TODO: prediction here, adding simulation delay for a while
    time.sleep(5)
    return {'next_hop': 'tbd'}, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
