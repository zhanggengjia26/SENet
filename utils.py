

from scipy.signal import butter, lfilter
def butter_bandpass_filter(data, fs, order=4):
    nyquist = 0.5 * fs
    low = 0.9 / nyquist
    high = 10 / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data, axis=-1)
    return y

def MinMax(ABPs):
    s = []
    d = []

    for i in range(0, len(ABPs)):
        max_value = max(ABPs[i])
        min_value = min(ABPs[i])

        s.append(min_value)
        d.append(max_value)
    return s,d