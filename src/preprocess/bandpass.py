import scipy.signal as sps


def bandpass_filter(x, fs, low=0.67, high=100.0, order=4):
    nyq = 0.5 * fs
    b, a = sps.butter(order, [low / nyq, high / nyq], btype='band')
    return sps.filtfilt(b, a, x)
