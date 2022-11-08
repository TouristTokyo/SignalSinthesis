import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfftfreq
from scipy.fft import rfft

amplitude = 1


def get_function(t, freq):
    return amplitude * np.sin(freq * t * 2 * np.pi)


def get_amplitude_modulation(t, carrier_freq, freq, k):
    return get_function(t, carrier_freq) * (amplitude + k * np.sign(np.sin(freq * t * 2 * np.pi)))


def get_frequency_modulation(t, carrier_freq, freq):
    freq_dop = 20
    result_values = []
    for time in t:
        if np.sign(get_function(time, freq)) > 0:
            result_values.append(get_function(time, carrier_freq))
        else:
            result_values.append(get_function(time, freq_dop))
    return result_values


def get_phase_modulation(t, carrier_freq, freq):
    result_values = []
    for time in t:
        if np.sign(get_function(time, freq)) > 0:
            result_values.append(get_function(time, carrier_freq))
        else:
            result_values.append(-get_function(time, carrier_freq))
    return result_values


def draw_amplitude_modulation(t, carrier_freq, freq, k):
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Amplitude modulation")
    amp_mod = get_amplitude_modulation(t, carrier_freq, freq, k)

    plt.subplot(2, 1, 1)
    plt.grid()
    plt.plot(t, amp_mod)

    plt.subplot(2, 1, 2)
    plt.grid()
    yf = rfft(amp_mod)
    yf[0] = 0
    xf = rfftfreq(len(t), 1 / len(t))
    plt.plot(xf, np.abs(yf))

    plt.show()


def draw_frequency_modulation(t, carrier_freq, freq):
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Frequency modulation")
    freq_mod = get_frequency_modulation(t, carrier_freq, freq)

    plt.subplot(2, 1, 1)
    plt.grid()
    plt.plot(t, freq_mod)

    plt.subplot(2, 1, 2)
    plt.grid()
    yf = rfft(freq_mod)
    yf[0] = 0
    xf = rfftfreq(len(t), 1 / len(t))
    plt.plot(xf, np.abs(yf))

    plt.show()


def draw_phase_modulation(t, carrier_freq, freq):
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Phase modulation")
    phase_mod = get_phase_modulation(t, carrier_freq, freq)

    plt.subplot(2, 1, 1)
    plt.grid()
    plt.plot(t, phase_mod)

    plt.subplot(2, 1, 2)
    plt.grid()
    yf = rfft(phase_mod)
    yf[0] = 0
    xf = rfftfreq(len(t), 1 / len(t))
    plt.plot(xf, np.abs(yf))

    plt.show()


def get_synthetic_signal(spectrum):
    harmonics = []
    for i in range(len(spectrum)):
        if np.abs(spectrum[i]) < 100:
            harmonics.append(0)
        else:
            harmonics.append(np.abs(spectrum[i]))

    signal = np.fft.ifft(np.array(harmonics))
    return signal


def get_filtered_signal(synthetic_signal, carrier_freq):
    filtered_signal = []
    level_sig = 1.4
    d = round(len(synthetic_signal) / carrier_freq)
    for k in range(len(synthetic_signal)):
        if synthetic_signal[k] > level_sig:
            filtered_signal.append(1)
        else:
            big_neighbour_found = False
            j = max(0, k - d)
            q = min(len(synthetic_signal) - 1, k + d)
            while j <= q:
                if synthetic_signal[j] > level_sig:
                    big_neighbour_found = True
                    break
                j += 1
            if big_neighbour_found:
                filtered_signal.append(1)
            else:
                filtered_signal.append(-1)
    return filtered_signal


def draw_synthesis_signal(t, carrier_freq, freq, k):
    amp_mod = get_amplitude_modulation(t, carrier_freq, freq, k)
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Synthesized signal")

    plt.grid()
    yf = np.abs(rfft(amp_mod))
    yf[0] = 0
    xf = rfftfreq(len(t), 1 / len(t))
    plt.plot(xf / (len(t) / 2), np.real(get_synthetic_signal(yf)))

    plt.show()


def draw_filter_signal(t, carrier_freq, freq, k):
    amp_mod = get_amplitude_modulation(t, carrier_freq, freq, k)
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Filtered signal")

    plt.grid()
    yf = np.abs(rfft(amp_mod))
    yf[0] = 0
    xf = rfftfreq(len(t), 1 / len(t))

    filter_signal = get_filtered_signal(np.abs(get_synthetic_signal(yf)), 100)
    plt.plot(xf / (len(t) / 2), filter_signal)

    plt.show()


def main():
    carrier_freq = 100
    freq = 10
    k = 0.5
    time = np.linspace(0, 1, num=900)
    draw_amplitude_modulation(time, carrier_freq, freq, k)
    draw_frequency_modulation(time, carrier_freq, freq)
    draw_phase_modulation(time, 30, 15)
    draw_synthesis_signal(time, carrier_freq, freq, k)
    draw_filter_signal(time, carrier_freq, freq, k)


if __name__ == '__main__':
    main()
