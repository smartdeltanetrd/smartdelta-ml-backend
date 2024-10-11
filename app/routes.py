from flask import Blueprint, jsonify, request
from app.data_utils import read_csv_to_dataframe, validate_and_get_file, validate_and_get_filePair
from app.MessageList import *
from app.trend_prediction import plot_case, generate_resource_load
import time
import logging
from io import StringIO
import pandas as pd
import os

bp = Blueprint('routes', __name__)

# Logging configuration
logging.basicConfig(level=logging.INFO)

from app.usecases import anomaly_train, next_hop_train, detect_anomalies



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

@bp.route('/categorical_comparison', methods=['POST'])
def categorical_comparison():
    try:
        file1, file2, error = validate_and_get_filePair(request)
        if error:
            return jsonify(message=error, error="Bad Request"), 400
        
        # Get the absolute path to the "app" folder
        app_folder = os.path.dirname(os.path.realpath(__file__))

       # Construct the absolute path to the "data" folder
        data_folder = os.path.join(app_folder, 'data')
        output_folder = os.path.join(app_folder, 'outputs')

# Construct the absolute paths to your CSV files

       # Convert file streams to DataFrames
        logs1_df = pd.read_csv('app/data/chatbroker.csv')
        logs2_df = pd.read_csv('app/data/smsbroker.csv')
        logs1_df = logs1_df[:len(logs2_df)]  # temporary hack

        print("Columns in logs1_df:", logs1_df.columns)
        print("Columns in logs2_df:", logs2_df.columns)


        msgs1 =  MessageList(logs1_df,'file1', verbose=True)
        msgs2 =  MessageList(logs2_df,'file2', verbose=True)
        #
        msgs1 = MessageList(logs1_df, 'app/data/chatbroker.csv', verbose=True)
        msgs2 = MessageList(logs2_df, 'app/data/smsbroker.csv', verbose=True)

		#
        stat1, cat_field_names1 = msgs1.categorical_combinations_stat()

        stat2, cat_field_names2 = msgs2.categorical_combinations_stat()
 
        cat_clusters1=msgs1.categorical_clusters_from_stat(stat1, cat_field_names1)

        cat_clusters2=msgs2.categorical_clusters_from_stat(stat2, cat_field_names2)
        

        json_clusters1 = msgs1.convert_categorical_clusters_to_json(cat_clusters1, cat_field_names1)
        json_clusters2 = msgs2.convert_categorical_clusters_to_json(cat_clusters2, cat_field_names2)
   
        res= compare_messages_by_categorical_fields(msgs1, msgs2, field_names=['Level', 'Method'])
        result_json = {
        'file1': json_clusters1,
        'file2': json_clusters2,
        'noise': res
         }

        return jsonify(result=result_json), 200

    except Exception as e:
        logging.error("Categorical Comparison endpoint failed: %s", str(e))
        return jsonify(message="An error occurred during training.", error=str(e)), 500

@bp.route('/anomaly_detection', methods=['POST'])
def anomaly_detection():
    try:
        data = request.json.get('data')  # Assuming the data is sent as JSON
        #print('Received data:', data)
        # Process the array data as needed
        anomalies = detect_anomalies(data)
        result = {'message': 'Log data received successfully and anomaly detection implemented', 'data': anomalies}
        return jsonify(result), 200
    except Exception as e:
        logging.error("Anomaly Detection endpoint failed: %s", str(e))
        return jsonify(message="An error occurred during detection.", error=str(e)), 500

@bp.route('/predict_resource', methods=['GET'])
def trend_prediction():
    try:
        GENERATED_DATA_PTS = 700
        WARNING_VALUE = 90

        res_times, res_vals = generate_resource_load(GENERATED_DATA_PTS, 'linear')
        
        result = plot_case(res_times, res_vals, WARNING_VALUE)
        print(result)
        return jsonify({
            'message': 'Trend prediction completed successfully',
            'result': result
        }), 200
    except Exception as e:
        return jsonify({
            'message': 'Error occurred during trend prediction',
            'error': str(e),
        }), 500
