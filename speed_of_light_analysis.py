import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Parameters
FILE_PATH = r"T:\Downloads\pmt_discrete_beam_20260512_175027.csv" # Path to oscilloscope .csv file
DISTANCE  = 1.5 # Distance from photodetector to mirror (one-way) [m]
SIGMA_D   = 0.001 # Distance uncertainty [m]
SIGMA_V   = 0.05 # Voltage uncertainty [V]
TIME_UNIT = 1e-9  # Unit of the time column in the CSV: 1.0 = seconds, 1e-9 = nanoseconds
C_LITERATURE = 299_792_458  # [m/s]
PROMINENCE = 0.1  # Minimum prominence for peak detection in the derivative

# Load .csv file (time (s), voltage (V))
df = pd.read_csv(FILE_PATH, header=None, comment="#")
df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
time = df.iloc[:, 0].to_numpy() * TIME_UNIT
voltage = df.iloc[:, 1].to_numpy()

# Calculate first derivative of the voltage signal
deriv = np.gradient(voltage, time)

# Find all prominent peaks in the derivative
min_distance = max(1, int(0.01 * len(deriv)))
peaks, _ = find_peaks(deriv, prominence=PROMINENCE, distance=min_distance)

if len(peaks) < 2:
    raise RuntimeError(f"Only {len(peaks)} peak(s) found.")

peak_times = time[peaks]
gaps = np.diff(peak_times)

# Intra-pair gaps (direct/reflected) are small; inter-pair gaps are large.
# Split using the midpoint between min and max gap as a threshold.
gap_threshold = (gaps.min() + gaps.max()) / 2

pair_deltas  = []
pair_indices = []  # (index_of_peak1, index_of_peak2) for each pair
i = 0
while i < len(peaks) - 1:
    if gaps[i] < gap_threshold:
        pair_deltas.append(peak_times[i + 1] - peak_times[i])
        pair_indices.append((peaks[i], peaks[i + 1]))
        i += 2
    else:
        i += 1

if len(pair_deltas) == 0:
    raise RuntimeError("No peak pairs found. Try adjusting the prominence threshold.")

delta_t = np.mean(pair_deltas)

# Timing uncertainty: std across pairs if multiple, else one sample interval
sigma_t = np.std(pair_deltas) if len(pair_deltas) > 1 else np.mean(np.diff(time))

# Use the first pair for plot annotations
top_two = np.array(pair_indices[0])
t1, t2  = time[top_two[0]], time[top_two[1]]

# Calculate speed of light and uncertainty
v = 2.0 * DISTANCE / delta_t
sigma_v = v * np.sqrt((SIGMA_D / DISTANCE) ** 2 + (sigma_t / delta_t) ** 2)

abs_error = abs(v - C_LITERATURE)
rel_error = abs_error / C_LITERATURE * 100

# Print results
print("")
print("--- Data " + "-" * 48)
print(f"Peak pairs found     : {len(pair_deltas)}")
print(f"Mean peak interval   : {delta_t*1e9:.4f} +/- {sigma_t*1e9:.4f} ns")
print("")
print("--- Results " + "-" * 45)
print("Experimental results:")
print(f"Speed of light       : {v:.4e} +/- {sigma_v:.2e} m/s")
print(f"Relative error       : {rel_error:.2f} %")
print("")
print("Comparison with literature:")
print(f"Literature           : {C_LITERATURE:.4e} m/s")
print(f"Discrepancy          : {abs_error:.2e} m/s  ({rel_error:.2f} %)")
print("")

# Plot
BG = '#0d0d1a'
TX = '#ccccdd'
peak_colors = ['#ffdd44', '#ff8844']

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, facecolor=BG)
fig.suptitle("Speed of Light - Oscilloscope Data", color='white', fontweight="bold", fontsize=11)

for ax in (ax0, ax1):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TX, labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_edgecolor('#334')

step = max(1, len(time) // 200)
ts   = time[::step] * 1e9
vs   = voltage[::step]
ds   = deriv[::step]

noise_v = SIGMA_V if SIGMA_V is not None else np.std(voltage[:max(1, len(voltage) // 10)])
noise_d = np.std(deriv[:max(1, len(deriv) // 10)])

ax0.errorbar(ts, vs, xerr=sigma_t * 1e9, yerr=noise_v,
             fmt='o', color="#66ccff", markersize=3,
             ecolor='#66ccff', elinewidth=0.6, capsize=2, alpha=0.8)
ax0.set_xlabel("Time (ns)", color=TX, fontsize=9)
ax0.set_ylabel("Voltage (V)", color=TX, fontsize=9)
ax0.grid(True, alpha=0.3)
ax0.set_title("Raw voltage signal", color=TX, fontsize=9.5, pad=4)

ax1.errorbar(ts, ds, xerr=sigma_t * 1e9, yerr=noise_d,
             fmt='o', color="#c77dff", markersize=3,
             ecolor='#c77dff', elinewidth=0.6, capsize=2, alpha=0.8)
for k, (idx1, idx2) in enumerate(pair_indices):
    for i, idx in enumerate([idx1, idx2]):
        color = peak_colors[i]
        label = ["direct light", "reflected light"][i] if k == 0 else None
        ax1.axvline(time[idx] * 1e9, color=color, linestyle="--", linewidth=1.4, alpha=0.85)
        ax1.plot(time[idx] * 1e9, deriv[idx], "o", color=color, markersize=8, label=label)

ax1.set_xlabel("Time (ns)", color=TX, fontsize=9)
ax1.set_ylabel("dV/dt (V/s)", color=TX, fontsize=9)
ax1.set_title("First derivative", color=TX, fontsize=9.5, pad=4)
leg = ax1.legend(fontsize=8, framealpha=0.20, facecolor='#1a1a2e')
for txt in leg.get_texts():
    txt.set_color(TX)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
