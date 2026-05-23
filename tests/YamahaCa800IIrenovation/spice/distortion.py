#! python3
import numpy as np
import subprocess


with open("thd.log", "w") as f:
    subprocess.run(
        ["ngspice", "-b", "thd.cir"],
        stdout=f,
        stderr=subprocess.STDOUT,  # optional: merge stderr into same file
        check=True
    )

with open("imd.log", "w") as f:
    subprocess.run(
        ["ngspice", "-b", "imd.cir"],
        stdout=f,
        stderr=subprocess.STDOUT,  # optional: merge stderr into same file
        check=True
    )

# ============================================================
# I/O
# ============================================================

def load_time(fname):
    data = np.loadtxt(fname)
    t = data[:, 0]          # time
    v = data[:, -1]         # signal (last column)
    return t, v

def crop_window(t, v, t0, t1):
    m = (t >= t0) & (t <= t1)
    return t[m], v[m]

def resample_uniform(t, v, fs):
    dt = 1.0 / fs
    tu = np.arange(t[0], t[-1], dt)
    vu = np.interp(tu, t, v)
    return tu, vu

# ============================================================
# FFT utilities
# ============================================================

def fft_vrms(t, v):
    dt = t[1] - t[0]
    N = len(v)

    v = v - np.mean(v)
    w = np.hanning(N)
    cg = np.mean(w)

    X = np.fft.rfft(v * w)
    f = np.fft.rfftfreq(N, dt)

    mag_pk = (2.0 / (N * cg)) * np.abs(X)
    mag_rms = mag_pk / np.sqrt(2)

    mag_rms[0] *= 0.5
    if N % 2 == 0:
        mag_rms[-1] *= 0.5

    return f, mag_rms

def band_rms(f, spec, f0, bw):
    h = bw / 2
    m = (f >= f0 - h) & (f <= f0 + h)
    return np.sqrt(np.sum(spec[m] ** 2)) if np.any(m) else 0.0

def band_total_rms(f, spec, f_lo, f_hi):
    m = (f >= f_lo) & (f <= f_hi)
    return np.sqrt(np.sum(spec[m] ** 2)) if np.any(m) else 0.0

def dbc(v, ref):
    return 20 * np.log10(v / ref) if v > 0 and ref > 0 else -np.inf

# ============================================================
# THD / THD+N
# ============================================================

def compute_thd_thdn(fname, t0, t1, fs, f0=1000.0, max_h=10):
    t, v = load_time(fname)
    t, v = crop_window(t, v, t0, t1)
    t, v = resample_uniform(t, v, fs)

    f, spec = fft_vrms(t, v)
    df = float(np.mean(np.diff(f)))
    BW = 5 * df

    V1 = band_rms(f, spec, f0, BW)
    harms = [band_rms(f, spec, n * f0, BW) for n in range(2, max_h + 1)]
    THD = np.sqrt(np.sum(np.square(harms))) / V1 if V1 > 0 else np.nan

    Vtot = band_total_rms(f, spec, 20.0, 20000.0)
    THDN = np.sqrt(max(Vtot**2 - V1**2, 0.0)) / V1 if V1 > 0 else np.nan

    return V1, THD, THDN

# ============================================================
# CCIF IMD / "IMD+N"
# ============================================================

def compute_ccif_imd_imdn(fname, t0, t1, fs, f1=19000.0, f2=20000.0,
                          imdn_band=(17000.0, 22000.0)):
    """
    Returns:
      IMD: classic CCIF using only 18k and 21k (3rd order)
      IMDN: "IMD+N-like" ratio in a band around the carriers excluding carriers
            (distortion+noise near carriers / carriers)
    """
    t, v = load_time(fname)
    t, v = crop_window(t, v, t0, t1)
    t, v = resample_uniform(t, v, fs)

    f, spec = fft_vrms(t, v)
    df = float(np.mean(np.diff(f)))
    BW = 5 * df  # integration bandwidth around tones

    # carriers
    V19 = band_rms(f, spec, f1, BW)
    V20 = band_rms(f, spec, f2, BW)
    Vcar = np.sqrt(V19**2 + V20**2)

    # classic CCIF IMD (3rd-order)
    V18 = band_rms(f, spec, 2*f1 - f2, BW)
    V21 = band_rms(f, spec, 2*f2 - f1, BW)
    IMD = np.sqrt(V18**2 + V21**2) / (V19 + V20) if (V19 + V20) > 0 else np.nan

    # "IMD+N": total energy in band near carriers, excluding carrier windows
    f_lo, f_hi = imdn_band
    Vband = band_total_rms(f, spec, f_lo, f_hi)

    # remove carrier contributions (power subtraction)
    Vex = np.sqrt(max(Vband**2 - V19**2 - V20**2, 0.0))

    IMDN = Vex / Vcar if Vcar > 0 else np.nan
    IMDN_dbc = dbc(Vex, Vcar)

    return (V19, V20, V18, V21, IMD, IMDN, IMDN_dbc, df, BW)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    FS = 500_000.0

    # steady-state windows matching your tran setups
    THD_T0, THD_T1 = 0.150, 0.200
    IMD_T0, IMD_T1 = 0.160, 0.320

    V1, THD, THDN = compute_thd_thdn("thd_time.csv", THD_T0, THD_T1, FS)

    print("==== THD / THD+N ====")
    print(f"Fundamental Vrms      = {V1:.6e}")
    print(f"THD                  = {THD*100:.6f} %")
    print(f"THD+N (20–20k)        = {THDN*100:.6f} %\n")

    V19, V20, V18, V21, IMD, IMDN, IMDN_dbc, df, BW = compute_ccif_imd_imdn(
        "imd_time.csv", IMD_T0, IMD_T1, FS, 19000.0, 20000.0,
        imdn_band=(17000.0, 22000.0)
    )

    print("==== CCIF IMD / IMD+N-like ====")
    print(f"FFT df               = {df:.6f} Hz   BW={BW:.6f} Hz")
    print(f"Carrier 19k Vrms      = {V19:.6e}")
    print(f"Carrier 20k Vrms      = {V20:.6e}")
    print(f"Carrier balance       = {V19/V20:.6f}")
    print(f"IMD 18k Vrms          = {V18:.6e}")
    print(f"IMD 21k Vrms          = {V21:.6e}")
    print(f"IMD (18k/21k)         = {IMD*100:.6f} %")
    print(f"IMD+N-like (17–22k)    = {IMDN*100:.6f} %")
    print(f"IMD+N-like             = {IMDN_dbc:.2f} dBc")

