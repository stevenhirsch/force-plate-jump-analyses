from scipy import signal


def butterworth_filter(arr, cutoff_frequency, fps, padding, order= 4):
    w = cutoff_frequency / (fps / 2)  # Normalize the frequency
    b, a = signal.butter(order, w, 'low')
    if len(arr) - 1 < fps:  # i.e., the length of the arr is less than 1 second
        output = signal.filtfilt(b, a, arr, method='gust')
    else:
        output = signal.filtfilt(b, a, arr, padlen=padding, method='pad')
    return output