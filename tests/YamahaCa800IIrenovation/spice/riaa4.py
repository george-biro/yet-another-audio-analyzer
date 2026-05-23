#! python3
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import argparse
import sys

class CustomHelpFormatter(argparse.HelpFormatter):
    def _get_help_string(self, action):
        help = action.help
        if action.default is not argparse.SUPPRESS:
            help += f' (default: {action.default})'
        return help

# Begin of the program
parser = argparse.ArgumentParser(
                    prog='Yet Another Audio Analyzer',
                    usage='%(prog)s [options]',
                    formatter_class=CustomHelpFormatter)

parser.add_argument("--acfile", type=str, default="sim_ac.csv")
parser.add_argument(
    "--param",
    action="append",
    default=[],
    help="ngspice parameter definition (passed as --define=...)"
)
args = parser.parse_args()

with open("acsim.log", "w") as f:
    subprocess.run(
        ["ngspice", "-b", "acsim.cir"],
        stdout=f,
        stderr=subprocess.STDOUT,  # optional: merge stderr into same file
        check=True
    )

# ============================================================
# Load ngspice wrdata CSV (Mac format: all vectors are complex)
#
# Columns:
# 0 : freq (real)
# 1 : Re(freq)   [ignore]
# 2 : Im(freq)   [ignore]
# 3 : freq (vout) [ignore]
# 4 : Re(Vout)
# 5 : Im(Vout)
# 6 : freq (vin)  [ignore]
# 7 : Re(Vin)
# 8 : Im(Vin)
# ============================================================
data = np.loadtxt(args.acfile)

freq = data[:, 0]                       # Hz
Vout = data[:, 4] + 1j * data[:, 5]
Vin  = data[:, 7] + 1j * data[:, 8]

# H = Vin / Vout
H = Vout / Vin

mag_db = 20 * np.log10(np.abs(H))
phase_deg = np.unwrap(np.angle(H)) * 180 / np.pi


# ============================================================
# Exact RIAA PLAYBACK (de-emphasis) transfer function
# ============================================================

# RIAA time constants:
# 3180 µs → 50.05 Hz
# 318  µs → 500.5 Hz
# 75   µs → 2122 Hz

f1 = 50.05     # Hz
f2 = 500.5     # Hz
f3 = 2122.0    # Hz

w = 2 * np.pi * freq
w1 = 2 * np.pi * f1
w2 = 2 * np.pi * f2
w3 = 2 * np.pi * f3

# Playback (DE-emphasis)
H_riaa = (1 + 1j*w/w2) / ((1 + 1j*w/w1) * (1 + 1j*w/w3))

mag_db_riaa = 20 * np.log10(np.abs(H_riaa))
phase_deg_riaa = np.unwrap(np.angle(H_riaa)) * 180 / np.pi


# ============================================================
# Normalize MAGNITUDE and PHASE at 1 kHz (RIAA reference)
# ============================================================

idx_1k = np.argmin(np.abs(freq - 1000))



# Magnitude normalization
mag_offset_db = mag_db[idx_1k] - mag_db_riaa[idx_1k]
mag_db_riaa_offset = mag_db_riaa + mag_offset_db

# Phase normalization
phase_offset = phase_deg[idx_1k] - phase_deg_riaa[idx_1k]
phase_deg_riaa_offset = phase_deg_riaa + phase_offset

mag_err = mag_db - mag_db_riaa_offset
phase_err = phase_deg - phase_deg_riaa_offset

print("Errors")
for f in [100, 500, 1000, 2000, 5000, 10000, 20000]:
    idx_f = np.argmin(np.abs(freq - f))
    print(f"{mag_err[idx_f]:8.2f}dB {phase_err[idx_f]:8.2f}deg @ {f:8.0f}Hz") 


# ============================================================
# Combined magnitude + phase figure
# ============================================================

fig, (ax_mag, ax_phase) = plt.subplots(
    2, 1, figsize=(11, 7), sharex=True
)

# ----------------------------
# Magnitude response subplot
# ----------------------------
ax_mag.semilogx(freq, mag_db, label="Simulated Circuit")
ax_mag.semilogx(
    freq, mag_db_riaa_offset, "--",
    label="RIAA Playback (aligned @ 1 kHz)"
)
ax_mag.set_ylabel("Magnitude [dB]")
ax_mag.set_title("Yamaha CA-800 II Phono – RIAA Playback Accuracy")
ax_mag.grid(True, which="both", ls="--")
ax_mag.legend()

# ----------------------------
# Phase response subplot
# ----------------------------
ax_phase.semilogx(freq, phase_deg, label="Simulated Circuit Phase")
ax_phase.semilogx(
    freq, phase_deg_riaa_offset, "--",
    label="RIAA Playback Phase (aligned @ 1 kHz)"
)
ax_phase.set_xlabel("Frequency [Hz]")
ax_phase.set_ylabel("Phase [degrees]")
ax_phase.set_title("Phase Response Comparison (1 kHz aligned)")
ax_phase.grid(True, which="both", ls="--")
ax_phase.legend()

plt.tight_layout()
plt.savefig("riaa.pdf")
plt.show()
