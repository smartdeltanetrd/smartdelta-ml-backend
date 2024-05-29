import logging
import time
import numpy as np
from app.LogAnalyzer import *


def detect_anomalies(logs):
    # TODO: use ml or another ds solution to categorize logs message then detect anomalies 
    log_analyzer = LogAnalyzer(logs)
    WIN_SIZE_SEC = 3
    N_PTS = 1000
  
    #anomaly_list = log_analyzer.report_anomalies('@timestamp', WIN_SIZE_SEC)

 
 
    noise = np.random.normal(loc=-8*WIN_SIZE_SEC, scale=5*WIN_SIZE_SEC, size=N_PTS)
    noise[noise < 0] = 0
    
    anomaly_list = log_analyzer.gen_data_and_report_result(noise,WIN_SIZE_SEC)
    sorted_anomalies = sorted(anomaly_list, key=lambda x: x['score'], reverse=True)
    return sorted_anomalies


def anomaly_train(df):
    if df.empty:
        logging.error("Anomaly training failed: Empty DataFrame.")
        return "Anomaly training failed: Empty DataFrame."
    print("ANOMALY TRAIN: DataFrame received.")
    print(df)
    # TODO: training here, adding simulation delay for a while
    time.sleep(5)
    return "Anomaly training completed successfully."


def next_hop_train(df):
    if df.empty:
        logging.error("Next-hop training failed: Empty DataFrame.")
        return "Next-hop training failed: Empty DataFrame."
    print("NEXTHOP TRAIN: DataFrame received.")
    print(df)
    # TODO: training here, adding simulation delay for a while
    time.sleep(5)
    processed_lines = 0
    return f"Next-hop training completed successfully. Processed lines: {processed_lines}"

