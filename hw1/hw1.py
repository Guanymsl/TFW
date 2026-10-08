import wave
import numpy as np

def gwave(a, b, c, T, Fs):
    N = int(round(T * Fs))
    t = np.arange(N) / Fs

    critical_times = [0.0, T]
    if a != 0:
        vertex = -b / (2 * a)
        if 0 < vertex < T:
            critical_times.append(vertex)

    max_frequency = max(
        abs(a * s**2 + b * s + c)
        for s in critical_times
    )

    if max_frequency >= Fs / 2:
        raise ValueError("Instantaneous frequency exceeds the Nyquist limit.")

    phase = 2 * np.pi * (
        a * t**3 / 3
        + b * t**2 / 2
        + c * t
    )

    signal = np.cos(phase)

    audio = np.round(signal * 32767).astype(np.int16)

    with wave.open("gwave.wav", "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(Fs)
        wav.writeframes(audio.tobytes())

    print("Saved gwave.wav")
    print(f"Duration: {N / Fs:.4f} seconds")
    print(f"Sampling frequency: {Fs} Hz")

if __name__ == "__main__":
    gwave(100, 200, 300, 5, 44100)
