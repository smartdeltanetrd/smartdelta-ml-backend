import pandas as pd
import numpy as np

class ChartData:
    def __init__(self, raw_data, mean_values, probability_distribution, predictionLine=None,predicted_warning=None, warning_distribution=None, WARNING_VALUE=90):
        self.raw_data = raw_data
        self.mean_values = mean_values
        self.probability_distribution = probability_distribution
        self.predicted_warning = predicted_warning
        self.warning_distribution = warning_distribution
        self.WARNING_VALUE = WARNING_VALUE
        self.predictionLine = predictionLine
        
        

    def to_dict(self):
        return {
            'raw_data': self.raw_data,
            'mean_values': self.mean_values,
            'probability_distribution': self.probability_distribution,
            'predicted_warning': self.predicted_warning,
            'warning_distribution': self.warning_distribution,
            'WARNING_VALUE': self.WARNING_VALUE,
            'prediction_line': self.predictionLine
            
            
        }
    


def generate_resource_load(max_idx, trend_type):
    lam = 1
    noise = 15*(np.random.poisson(lam, max_idx + 1) - lam)/np.sqrt(lam)
    time_idx = np.sort(np.random.choice(a=list(range(0, max_idx)), size=np.random.randint(max_idx//5, max_idx), replace=False))
    time_idx = np.append(time_idx, max_idx)
    if trend_type == 'linear':
        trend = 10 + time_idx*(70 - 10)/max_idx
    elif trend_type == 'const':
        trend = 30
    elif trend_type == 'exp':
        trend = 3 + np.exp(4*time_idx/max_idx)*(70 - 3)/np.exp(4*max_idx/max_idx)
    else:
        raise Exception(f"Unknown trend type: {trend_type}")
    out_signal = trend + noise[time_idx]
    out_signal = np.clip(out_signal, 0, 100)
    return time_idx, out_signal
def preprocess_data(res_times, res_vals, stat_window):
    win_pos = res_times.max() - stat_window  # go from end to start, as end points are more important
    out_times = []
    out_means = []
    out_probs = []
    out_pdfs = []
    while win_pos >= 0:
        idx = np.argwhere((res_times >= win_pos) & (res_times < win_pos + stat_window))
        if len(idx) == 0:
            win_pos -= stat_window
            continue
        times = res_times[idx]
        vals = res_vals[idx]

        out_times.insert(0, win_pos + stat_window//2)
        out_means.insert(0, vals.mean())

        PDF_BINS = 20
        PTS_COUNT = len(vals)
        if PTS_COUNT > 0:
            # https://en.wikipedia.org/wiki/Standard_error#Finite_population_correction_(FPC)
            MIN_SAMPLES_PER_BIN = 20  # 5% error per bin?
            DESIRED_SAMPLES = PDF_BINS*MIN_SAMPLES_PER_BIN
            if PTS_COUNT < DESIRED_SAMPLES:
                fpc = np.sqrt((DESIRED_SAMPLES - PTS_COUNT)/(DESIRED_SAMPLES - 1))
            else:
                fpc = 0

            pdf_win_pos = vals.min()
            vals_max = vals.max()
            pdf_win_len = (vals_max + 1e-4 - vals.min())/PDF_BINS  # +1e-4 is to include vals_max value in bins
            hist = []
            while pdf_win_pos <= vals_max:
                k = np.count_nonzero((vals >= pdf_win_pos) & (vals < pdf_win_pos + pdf_win_len))
                hist.append(k/PTS_COUNT)
                pdf_win_pos += pdf_win_len
            out_probs.insert(0, 1 - fpc)  # maybe, it is a hack, but I don't know how to use FPC to estimate probability
            out_pdfs.insert(0, (np.arange(vals.min(), vals_max + pdf_win_len + 1e-4, pdf_win_len)[:len(hist) + 1], np.array(hist)))
        else:
            out_probs.insert(0, 0)
            out_pdfs.insert(0, None)

        win_pos -= stat_window

    out_times = np.array(out_times)
    out_means = np.array(out_means)
    out_probs = np.array(out_probs)
    remove_mask = out_probs < 1e-4
    out_times = out_times[~remove_mask]
    out_means = out_means[~remove_mask]
    out_probs = out_probs[~remove_mask]
    remove_idxs = np.argwhere(remove_mask)
    out_pdfs = [elem for i, elem in enumerate(out_pdfs) if i not in remove_idxs]
    return out_times, out_means, out_probs, out_pdfs
def predict_warning(mean_times, mean_vals, mean_probs, window_size, times, vals, warn_val):
    LAST_MEAN_PTS = 10  # process no more than last 10 mean points
    MAX_PREDICT_HORIZON_FACTOR = 10  # MAX_PREDICT_HORIZON_FACTOR*(mean_times.max() - mean_times.min())
    MAX_ABSOLUTE_ERROR_THR = 60  # useless? but less values can break functionality

    warn_time = None
    forecast_times = None
    forecast_vals = None
    t_pdf_xs = None
    t_pdf_ys = None

    # TODO: maybe we can predict, basion on only some last points and choose result by error
    if len(mean_times) > 0:
        # prob_coeffs = np.linspace(0, 1, LAST_MEAN_PTS + 1)[1:]  # take last point with max weight, decrease linearly
        # new_probs = mean_probs[-LAST_MEAN_PTS:]*prob_coeffs[-min(len(mean_probs), LAST_MEAN_PTS):]
        new_probs = mean_probs[-LAST_MEAN_PTS:]
        lin_coeffs = np.polyfit(mean_times[-LAST_MEAN_PTS:], mean_vals[-LAST_MEAN_PTS:], 1, w=new_probs)
    else:
        return warn_time, forecast_times, forecast_vals, t_pdf_xs, t_pdf_ys

    if abs(lin_coeffs[0]) < 1e-4 or lin_coeffs[0] < 0:
        return warn_time, forecast_times, forecast_vals, t_pdf_xs, t_pdf_ys

    if abs(mean_vals[-LAST_MEAN_PTS:] - (lin_coeffs[0]*mean_times[-LAST_MEAN_PTS:] + lin_coeffs[1])).mean() > MAX_ABSOLUTE_ERROR_THR:
        return warn_time, forecast_times, forecast_vals, t_pdf_xs, t_pdf_ys

    warn_time = (warn_val - lin_coeffs[1])/lin_coeffs[0]
    if warn_time < mean_times.max() or warn_time > mean_times.min() + MAX_PREDICT_HORIZON_FACTOR*(mean_times.max() - mean_times.min()):
        warn_time = None
        return warn_time, forecast_times, forecast_vals, t_pdf_xs, t_pdf_ys
    else:
        forecast_times = np.linspace(mean_times.max(), warn_time, 10)
        forecast_vals = lin_coeffs[1] + lin_coeffs[0]*forecast_times

    recent_mask = times >= mean_times[-LAST_MEAN_PTS if len(mean_times) >= LAST_MEAN_PTS else 0] - window_size/2
    recent_times = times[recent_mask]
    recent_vals = vals[recent_mask]
    noise_vals = recent_vals - (lin_coeffs[0]*recent_times + lin_coeffs[1])

    PDF_BINS = 50
    pts_count = len(noise_vals)
    pdf_win_pos = noise_vals.min()
    vals_max = noise_vals.max()
    pdf_win_len = (vals_max + 1e-4 - noise_vals.min())/PDF_BINS  # +1e-4 is to include vals_max value in bins
    hist = []
    while pdf_win_pos <= vals_max:
        k = np.count_nonzero((noise_vals >= pdf_win_pos) & (noise_vals < pdf_win_pos + pdf_win_len))
        hist.append(k/pts_count)
        pdf_win_pos += pdf_win_len
    noise_pdf_xs = np.arange(noise_vals.min(), vals_max + pdf_win_len + 1e-4, pdf_win_len)[:len(hist) + 1]
    noise_pdf_ys = np.array(hist)

    t_vals = []
    t_probs = []
    for noise_val, noise_prob in zip(noise_pdf_xs, noise_pdf_ys):
        t_vals.append((warn_val - noise_val - lin_coeffs[1])/lin_coeffs[0])
        t_probs.append(noise_prob)
    t_vals = np.array(t_vals)
    t_probs = np.array(t_probs)

    PDF_BINS = 10
    pts_count = len(t_vals)
    pdf_win_pos = t_vals.min()
    vals_max = t_vals.max()
    pdf_win_len = (vals_max + 1e-4 - t_vals.min())/PDF_BINS  # +1e-4 is to include vals_max value in bins
    hist = []
    while pdf_win_pos <= vals_max:
        window_mask = (t_vals >= pdf_win_pos) & (t_vals < pdf_win_pos + pdf_win_len)
        if np.count_nonzero(window_mask) > 0:
            mean_prob = t_probs[window_mask].mean()
        else:
            mean_prob = 0
        hist.append(mean_prob)
        pdf_win_pos += pdf_win_len
    t_pdf_xs = np.arange(t_vals.min(), vals_max + pdf_win_len + 1e-4, pdf_win_len)[:len(hist) + 1]
    t_pdf_ys = np.array(hist)

    return warn_time, forecast_times, forecast_vals, t_pdf_xs, t_pdf_ys

def plot_case(res_times, res_vals, high_warn_level):
   
    time_avg_step = (res_times.max() - res_times.min()) / len(res_times)
    time_window_length = min(max((res_times.max() - res_times.min()) / 10, time_avg_step * 50),
                            time_avg_step * 200)
    prep_times, prep_means, prep_probs, prep_pdfs = preprocess_data(res_times, res_vals, time_window_length)
    warn_time, fore_times, fore_vals, t_pdf_xs, t_pdf_ys = predict_warning(
        prep_times, prep_means, prep_probs, time_window_length, res_times, res_vals, high_warn_level)

    chart_data = ChartData(
        raw_data={'times': res_times.tolist(), 'values': res_vals.tolist()},
        mean_values={'times': prep_times.tolist(), 'values': prep_means.tolist()},
        probability_distribution={'times': prep_times.tolist(), 'values': prep_probs.tolist()},
        predicted_warning=None,
        warning_distribution=None,
        predictionLine = {'times': fore_times.tolist(), 'values': fore_vals.tolist()}
    )

    # Update the ChartData instance based on the conditions
    if warn_time is not None:
        chart_data.predicted_warning = {'time': warn_time, 'values': fore_vals.tolist()}
        warning_distribution = {'times': t_pdf_xs[:-1].tolist(), 'values': t_pdf_ys.tolist()}
        chart_data.warning_distribution = warning_distribution

    return chart_data.to_dict()


GENERATED_DATA_PTS = 700

WARNING_VALUE = 90
res_times, res_vals = generate_resource_load(GENERATED_DATA_PTS, 'linear')
result = plot_case(res_times, res_vals, WARNING_VALUE)

# Print or inspect the result
print(result)


"""res_df = pd.read_csv('data/resource_load_dataset/4802498578')
res_times = (res_df.iloc[:, 0] - res_df.iat[0, 0]).to_numpy()/1e7
assert len(np.unique(res_times)) == len(res_times)
res_vals = np.clip(res_df.iloc[:, 2].to_numpy()*100, 0, 100)
assert len(res_vals) == len(res_times)

remove_mask = ~(np.isfinite(res_times) & np.isfinite(res_vals))
res_times = res_times[~remove_mask]
res_vals = res_vals[~remove_mask]

print(f"Data length: {len(res_times)}")
plot_case(res_times, res_vals, WARNING_VALUE)
"""