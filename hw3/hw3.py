import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from scipy.fft import fft, ifft, next_fast_len
from time import perf_counter

def Gabor(x, tau, t, f, sgm):
    x = np.asarray(x, dtype=np.complex128)
    tau = np.asarray(tau, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)

    N = len(x)
    dtau = tau[1] - tau[0]

    radius = np.sqrt(-np.log(1e-8) / (np.pi * sgm))
    M = int(np.ceil(radius / dtau))

    L = next_fast_len(N + 2 * M)
    X = fft(x, n=L)

    freq_bins = np.fft.fftfreq(L, d=dtau)

    positions = np.rint((t - tau[0]) / dtau).astype(int)
    valid = (positions >= 0) & (positions < N)

    y = np.zeros((len(f), len(t)), dtype=np.complex128)

    normalization = sgm**(-0.25)

    for i, freq in enumerate(f):
        diff = (freq_bins - freq + 0.5 / dtau) % (1 / dtau) - 0.5 / dtau

        H = np.exp(-np.pi * diff**2 / sgm)
        filtered = ifft(X * H)

        y[i, valid] = normalization * filtered[positions[valid]] * np.exp(-2j * np.pi * freq * t[valid])

    return y

def main():
    a1, fs = sf.read("Chord.wav")

    x = a1[:, 0] if a1.ndim > 1 else a1
    tau = np.arange(len(x)) / fs

    dt = 0.01
    df = 1
    sgm = 200

    t = np.arange(0, tau[-1] + dt / 2, dt)
    f = np.arange(20, 1000 + df, df)

    start = perf_counter()
    y = Gabor(x, tau, t, f, sgm)
    elapsed = perf_counter() - start

    print(f"Running time: {elapsed:.4f} seconds")

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, np.abs(y), shading="auto", cmap="jet")

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Scaled Gabor Transform (Unbalanced Form)")
    plt.colorbar(label="Magnitude")

    plt.tight_layout()
    plt.savefig("gabor.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
