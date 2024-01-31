import logging
import time


def detect_anomalies(logs):
    # TODO: use ml or another ds solution to categorize logs message then detect anomalies 
    time.sleep(2)
         # Dummy array of objects
    dummy_anomalies = [
            {
                'anomaly_score': 85,
                'anomaly': 'Fewer log messages',
                'start_time': '2024-01-31T12:00:00',
                'dataSet': 'unknown'
            },
           {
                'anomaly_score': 99,
                'anomaly': 'Fewer log messages',
                'start_time': '2024-01-31T12:00:00',
                'dataSet': 'unknown'
            },
           
        ]
    sorted_anomalies = sorted(dummy_anomalies, key=lambda x: x['anomaly_score'], reverse=True)

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

