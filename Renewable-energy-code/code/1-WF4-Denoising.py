import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
# Reading data
data = pd.read_csv('../Experimental data/WF4-66_filled_pre.csv')
power_data = data['Power (MW)'].values
# 1. DWT decomposition
def dwt_decomposition(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs
# 2. DFT denoising (for high frequency components)
def dft_denoise(high_freq_component, cutoff_freq=0.1):
    fft_signal = np.fft.fft(high_freq_component)
    frequencies = np.fft.fftfreq(len(high_freq_component))

    transfer_function = np.zeros_like(frequencies)
    transfer_function[np.abs(frequencies) < cutoff_freq] = 1

    fft_signal_filtered = fft_signal * transfer_function
    denoised_signal = np.fft.ifft(fft_signal_filtered)
    return np.real(denoised_signal)

# 3. Hybrid denoising
def hybrid_denoise(signal, wavelet='db4', level=3, cutoff_freq=0.1):
    coeffs = dwt_decomposition(signal, wavelet, level)

    for i in range(1, len(coeffs)):
        coeffs[i] = dft_denoise(coeffs[i], cutoff_freq)

    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal[:len(signal)]

denoised_power = hybrid_denoise(power_data)

# 4. Visualizing the results
plt.figure(figsize=(12, 6))
plt.plot(power_data, label='Original Power Data', alpha=0.5)
plt.plot(denoised_power, label='Hybrid (DWT + DFT) Denoised Power Data', linewidth=2)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Power (MW)')
plt.title('Power Data Before and After Denoising')
plt.show()

# Save the denoised data
data['denoised_power'] = denoised_power
data.to_csv('../Result-file/WF4-66-denoised.csv', index=False)
