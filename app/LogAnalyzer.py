import json
import datetime
import re
import pandas as pd
import math
from flask import Blueprint
bp = Blueprint('LogAnalyzer', __name__)
import numpy as np

class LogAnalyzer:
    def __init__(self, logs):
        self.logs_dict = logs

    def get_log_events(self, ts_win_address, win_size_sec):
        ts_vals = self.logs_dict
        print(ts_vals)
        for hop in ts_win_address.split('.'):
            new_ts_vals = []
            for msg in ts_vals:
                if hop not in msg:
                    print(f"WARNING: didn't find \"{hop}\" in message {msg}")
                else:
                    new_ts_vals.append(msg[hop])
            ts_vals = new_ts_vals

        if re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z', ts_vals[0]):
            # full timestamp with milliseconds
            format_str = '%Y-%m-%dT%H:%M:%S.%fZ'
        elif re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', ts_vals[0]):
            # timestamp with missed milliseconds
            format_str = '%Y-%m-%dT%H:%M:%SZ'
        else:
            raise Exception(f"ERROR: unknown time format: {ts_vals[0]}")
        ts_vals = sorted([
            datetime.datetime.strptime(ts, format_str)
            for ts in ts_vals
        ])  # automatic ISO conversion with "fromisoformat()" is supported only from Python 3.11

        idx_freq = max(win_size_sec // 2, 1)
        event_idx = pd.date_range(
            start=ts_vals[0],
            end=ts_vals[-1] + datetime.timedelta(seconds=1),
            freq=f'{idx_freq}s',
        )
        event_cnt = []
        idx = 0
        prev_time = None
        for next_time in event_idx:
            if prev_time is None:
                prev_time = next_time
                continue
            cnt = 0
            while idx < len(ts_vals) and ts_vals[idx] >= prev_time and ts_vals[idx] < next_time:
                cnt += 1
                idx += 1
            event_cnt.append(cnt)
            prev_time = next_time

        return pd.Series(data=event_cnt, index=event_idx[:-1])

    def find_event_time_anomaly(self, ts_win_address, win_size_sec):
        def aggregate_rolling_window_results(rolling_scores, threshold, win_size):
            # remove overlapped and consequtive detected windows from the list and output only (start, end)
            # intervals for the detected anomalies
            detected_win_starts = rolling_scores[rolling_scores > threshold].index
            if len(detected_win_starts) == 0:
                return [], []
            detected_wins = detected_win_starts.to_series(index=pd.RangeIndex(len(detected_win_starts)))
            bool_mask = [True] + (
                        detected_wins.diff(1).dropna() > pd.Timedelta(win_size, 's')).values.tolist()
            win_starts = detected_wins[bool_mask].index.tolist()
            win_size_td = datetime.timedelta(seconds=win_size)
            win_size_eps = datetime.timedelta(seconds=win_size / 2)
            out_windows = []
            out_scores = []
            for w_idx in range(len(win_starts) - 1):
                out_windows.append(
                    (detected_wins[win_starts[w_idx]],
                     detected_wins[win_starts[w_idx + 1] - 1] + win_size_td)
                )
                out_scores.append(
                    rolling_scores[detected_wins[win_starts[w_idx]]:detected_wins[win_starts[w_idx + 1] - 1] + win_size_eps]
                    .mean()
                )
            out_windows.append(
                (detected_wins[win_starts[-1]],
                 detected_wins.iat[-1] + win_size_td)
            )  # last window in series
            out_scores.append(
                rolling_scores[detected_wins[win_starts[-1]]:detected_wins.iat[-1] + win_size_eps]
                .mean()
            )
            return out_windows, out_scores

        ts_s = self.get_log_events(ts_win_address, win_size_sec)
        freqs = ts_s.rolling(window=f'{win_size_sec}s', min_periods=0).sum()
        freqs.index -= pd.Timedelta(win_size_sec, 's')  # to make window position on left side, not right one

        if len(freqs[freqs == 0]) >= 0.5 * len(freqs):
    # heuristic: when logs contain too many "silence periods", we don't analyze them at all,
    # otherwise they will make detector to mark almost any activity period as anomaly,
    # which is usually not true for the server with sporadic load
          freqs = freqs[freqs != 0]


        # Tukey's Fence
        k = 3  # to be sure that value is enough "far out"
        q1 = freqs.quantile(0.25)
        q3 = freqs.quantile(0.75)
        # low_thr = q1 - k * (q3 - q1)
        # high_thr = q3 + k * (q3 - q1)

        if q3 - q1 < 1e-8:
            return []

        too_rare_scores = (q1 - freqs) / (q3 - q1)  # the more the farer current value
        too_frequent_scores = (freqs - q3) / (q3 - q1)  # the more the farer current value

        too_rare_wins, too_rare_scores = aggregate_rolling_window_results(too_rare_scores, k, win_size_sec)
        too_frequent_wins, too_frequent_scores = aggregate_rolling_window_results(too_frequent_scores, k, win_size_sec)

        anomaly_objects = []
        for (win_start, win_end), score in zip(too_rare_wins, too_rare_scores):
            normed_score = (score - k) / k + 3  # 3 will give ~0.953 value below, so all anomalies will start from this probability
            anomaly_objects.append({
                'type': 'too_rare',
                'source_field': ts_win_address,
                'stationary_period_sec': win_size_sec,
                'message': "Fewer log messages",
                'win_start': win_start,
                'win_end': win_end,
                'score': int(100 / (1 + math.exp(-normed_score)))
            })
        for (win_start, win_end), score in zip(too_frequent_wins, too_frequent_scores):
            normed_score = (score - k) / k + 3  # 3 will give ~0.953 value below, so all anomalies will start from this probability
            anomaly_objects.append({
                'type': 'too_frequent',
                'source_field': ts_win_address,
                'stationary_period_sec': win_size_sec,
                'message': "More frequent log messages",
                'win_start': win_start,
                'win_end': win_end,
                'score': int(100 / (1 + math.exp(-normed_score)))
            })

        return anomaly_objects
    
    def gen_data_and_report_result(self,noise,win_size_sec):
       msg_secs = np.cumsum(noise)
       msg_times = [datetime.datetime.now() + datetime.timedelta(seconds=msg_sec) for msg_sec in msg_secs]

       gen_logs_dict = []
       for msg_time in msg_times:
           gen_logs_dict.append({'data': None, 'ts': msg_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')})
       self.logs_dict = gen_logs_dict
       return self.report_anomalies( 'ts', win_size_sec)

    def report_anomalies(self, ts_win_address, win_size_sec):
        print("SELF LOGS DICT:")
        print(self.logs_dict)
        anomaly_list = self.find_event_time_anomaly(ts_win_address, win_size_sec)

        if len(anomaly_list) > 0:
            for a_info in anomaly_list:
                if a_info['type'] == 'too_rare':
                    s = "Too few"
                elif a_info['type'] == 'too_frequent':
                    s = "Too frequent"
                else:
                    raise Exception(f"Unknown anomaly type: {a_info['type']}")
                print(f"{s}, from {a_info['win_start']} to {a_info['win_end']}")
                return anomaly_list
        else:
            print("No anomalies found")
            return []

# Example usage:
