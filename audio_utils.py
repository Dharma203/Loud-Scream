import numpy as np

def get_pitch_autocorrelation(audio_data, rate=44100):
    audio_data = audio_data - np.mean(audio_data)  # Hilangkan DC offset
    corr = np.correlate(audio_data, audio_data, mode='full')  # Autokorelasi
    corr = corr[len(corr)//2:]  # Ambil separuh bagian positif

    d = np.diff(corr)  # Turunan pertama
    start = np.where(d > 0)[0]  # Titik awal peningkatan pertama

    if len(start) == 0:
        return 0

    peak = np.argmax(corr[start[0]:]) + start[0]  # Temukan puncak korelasi

    return rate / peak if peak else 0  # Hitung pitch (frekuensi)
