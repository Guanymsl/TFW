import time
import numpy as np
import matplotlib.pyplot as plt

def recSTFT(x, t, f, B):
    x = np.asarray(x)
    t = np.asarray(t)
    f = np.asarray(f)

    dt = t[1] - t[0]
    df = f[1] - f[0]

    N = round(1 / (dt * df))

    Q = int(np.ceil(B / dt))
    offsets = np.arange(-Q, Q + 1)
    mask = np.abs(offsets * dt) < B

    bins = np.rint(f * N * dt).astype(int)

    y = np.zeros((len(t), len(f)), dtype=complex)

    for n in range(len(t)):
        start = n - Q
        indices = start + np.arange(len(offsets))

        segment = np.zeros(N, dtype=complex)

        valid = (
            (indices >= 0)
            & (indices < len(x))
            & mask
        )

        segment[np.where(valid)[0]] = x[indices[valid]]
        spectrum = np.fft.fft(segment)

        phase = np.exp(-2j * np.pi * f * start * dt)
        y[n, :] = dt * phase * spectrum[bins % N]

    return y

dt = 0.05
df = 0.05

t1 = np.arange(0, 10, dt)
t2 = np.arange(10, 20, dt)
t3 = np.arange(20, 30 + dt / 2, dt)

t = np.arange(0, 30 + dt / 2, dt)
f = np.arange(-5, 5 + df / 2, df)

x = np.concatenate([
    np.cos(2 * np.pi * t1),
    np.cos(6 * np.pi * t2),
    np.cos(4 * np.pi * t3)
])

B = 1.0

start_time = time.perf_counter()
y = recSTFT(x, t, f, B)
elapsed = time.perf_counter() - start_time

plt.figure(figsize=(10, 3))
plt.plot(t, x)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Input Signal", fontweight="bold")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.pcolormesh(t, f, np.abs(y).T, shading="auto")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title(
    "FFT-based Rectangular STFT",
    fontweight="bold",
    pad=28
)
plt.text(
    0.5, 1.02,
    f"Running time: {elapsed:.6f} s",
    transform=plt.gca().transAxes,
    ha="center",
    fontsize=10
)
plt.colorbar(label="Magnitude")
plt.tight_layout()
plt.show()
