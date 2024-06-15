import numpy as np
from scipy.signal import savgol_filter

def msc(input_data, mean_spectrum):
    corrected_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        fit = np.polyfit(mean_spectrum, input_data[i, :], 1, full=True)
        corrected_data[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]

    return corrected_data

def snv(input_data, mean_spectrum_snv, mean_spectrum_std):
    # Convert input to numpy arrays if they are not already
    input_data = np.asarray(input_data)
    if mean_spectrum_snv is None or mean_spectrum_std is None:
        mean_spectrum_snv = np.mean(input_data, axis=0, keepdims=True)
        mean_spectrum_std = np.std(input_data, axis=0, keepdims=True)
    #mean_spectrum_snv = np.asarray(mean_spectrum_snv)
    
    # Print shapes of input data and mean spectrum for verification
    print('input', input_data.shape)
    print('mean spectrum', mean_spectrum_snv.shape)
    print('mean spectrum + std', mean_spectrum_std.shape)
    
    # Perform SNV transformation
    if np.any(mean_spectrum_std == 0):
        raise ValueError("At least one spectrum has zero standard deviation, cannot perform SNV")
    
    snv_transformed = (input_data - mean_spectrum_snv) / mean_spectrum_std
    
    return snv_transformed, mean_spectrum_snv, mean_spectrum_std

def first_derivative(input_data, window_size=7, poly_order=2):
    return savgol_filter(input_data, window_length=window_size, polyorder=poly_order, deriv=1)

def second_derivative(input_data, window_size=5, poly_order=2):
    return savgol_filter(input_data, window_length=window_size, polyorder=poly_order, deriv=2)

def smoothing(input_data, window_size=5, poly_order=2):
    return savgol_filter(input_data, window_length=window_size, polyorder=poly_order)

def mean_centering(input_data, mean_spectrum):
    centered_data = input_data - mean_spectrum
    return centered_data

def normalize_spectral_data(data, max_vals, min_vals):
    # Ensure input is a numpy array
    data = np.asarray(data)
    
    print('data', data.shape)

    # Find the minimum and maximum values for each spectrum
    if max_vals is None or min_vals is None:
        min_vals = data.min(axis=0, keepdims=True)
        max_vals = data.max(axis=0, keepdims=True)
    
    # Normalize data to the range [0, 1]
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    
    return normalized_data, max_vals, min_vals