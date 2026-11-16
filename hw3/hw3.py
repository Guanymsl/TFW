import time
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def Gabor(x, tau, t, f, sgm):
    x = np.asarray(x, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)

    N = len(x)
    dtau = tau[1] - tau[0]

    y = np.zeros((len(f), len(t)), dtype=np.complex128)

    L = 1 << (3 * N - 2).bit_length()
    X = np.fft.fft(x, n=L)

    lag = np.arange(-N + 1, N) * dtau
    positions = np.rint((t - tau[0]) / dtau).astype(int)
    valid = (positions >= 0) & (positions < N)

    for i, freq in enumerate(f):
        sigma = sgm / abs(freq)

        window = np.exp(-lag**2 / (2 * sigma**2))
        window /= np.sqrt(2 * np.pi) * sigma

        kernel = window * np.exp(2j * np.pi * freq * lag)

        H = np.fft.fft(kernel, n=L)
        conv = np.fft.ifft(X * H)

        result = conv[N - 1:2 * N - 1] * dtau

        y[i, valid] = (
            result[positions[valid]]
            * np.exp(-2j * np.pi * freq * t[valid])
        )

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

    start = time.perf_counter()
    y = Gabor(x, tau, t, f, sgm)
    elapsed = time.perf_counter() - start

    print(f"Running time: {elapsed:.4f} seconds")
    print(f"Output shape: {y.shape}")

    plt.figure(figsize=(12, 6))
    plt.pcolormesh(
        t, f, np.abs(y),
        shading="auto",
        cmap="jet"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Scaled Gabor Transform (Unbalanced Form)")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.savefig("gabor.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
