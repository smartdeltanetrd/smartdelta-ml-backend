import logging
import time

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
