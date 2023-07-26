from flask import Blueprint, jsonify, request
from app.data_utils import read_csv_to_dataframe, validate_and_get_file
import time
import logging

bp = Blueprint('routes', __name__)

# Logging configuration
logging.basicConfig(level=logging.INFO)

from app.usecases import anomaly_train, next_hop_train


@bp.route('/train', methods=['POST'])
def train():
    try:
        f, error = validate_and_get_file(request)
        if error:
            return jsonify(message=error, error="Bad Request"), 400

        df = read_csv_to_dataframe(f)
        result_anomaly = anomaly_train(df)
        result_next_hop = next_hop_train(df)

        return jsonify(message_anomaly=result_anomaly, message_next_hop=result_next_hop), 200

    except Exception as e:
        logging.error("Train endpoint failed: %s", str(e))
        return jsonify(message="An error occurred during training.", error=str(e)), 500


@bp.route('/anomaly_predict', methods=['POST'])
def anomaly_predict():
    # TODO: prediction here, adding simulation delay for a while
    time.sleep(5)
    return {'anomalies': []}, 200


@bp.route('/next_hop_predict', methods=['POST'])
def next_hop_predict():
    # TODO: prediction here, adding simulation delay for a while
    time.sleep(5)
    return {'next_hop': 'tbd'}, 200
