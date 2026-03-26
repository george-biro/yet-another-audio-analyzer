#! /usr/bin/env python3
#
# Yet Another Audio analyzeR
#
# Copyright 2026 George Biro
#
# GPLv3-or-later
#
# File yaar.py
#
# MAC -> midi.app -> select E1DA -> set primary to zero

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass

from audio_backend import (
    open_stream,
    read_measurement,
    list_sound_devices,
    AudioConfig,
    RingBuffer,
)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter

plt.ion()
EPS = 1e-20


def simulate_signal(
    freq_hz: float,
    noise_amp: float,
    ts: np.ndarray,
    freq2_hz: float,
    amp2: float,
    hmncs: float,
    thd_hmncs: int,
) -> np.ndarray:

    phase1 = random.random() * 2 * np.pi
    sig1 = np.sin(2 * np.pi * freq_hz * ts + phase1)
    signal = sig1.copy()

    if freq2_hz > 0:
        phase2 = random.random() * 2 * np.pi
        sig2 = amp2 * np.sin(2 * np.pi * freq2_hz * ts + phase2)
        signal += sig2

    if hmncs > 0:

        dist = np.zeros_like(ts)

        # harmonic amplitude shape
        weights = np.array([1.0 / h for h in range(2, thd_hmncs + 1)])
        norm = np.sqrt(np.sum(weights**2))
        weights /= norm

        # harmonics of tone1

        if freq2_hz <= 0:
            # THD case
            for i, h in enumerate(range(2, thd_hmncs + 1)):
                phase = random.random() * 2 * np.pi
                dist += weights[i] * np.sin(2 * np.pi * (freq_hz * h) * ts + phase)
        else:
            # IMD products (normalized like harmonics)
            imd_freqs = [
                abs(freq2_hz - freq_hz),
                freq2_hz + freq_hz,
                abs(2 * freq_hz - freq2_hz),
                abs(2 * freq2_hz - freq_hz),
            ]

            imd_weights = np.ones(len(imd_freqs))
            imd_weights /= np.sqrt(np.sum(imd_weights**2))

            for w, f in zip(imd_weights, imd_freqs):
                phase = random.random() * 2 * np.pi
                dist += w * np.sin(2 * np.pi * f * ts + phase)

        # --- scale THD distortion ---
        p_sig = np.dot(signal, signal)
        p_dist = np.dot(dist, dist)
        dist *= hmncs * np.sqrt(p_sig / p_dist)
        signal += dist

    if noise_amp > 1e-6:
        signal += np.random.normal(0.0, noise_amp, len(signal))

    return signal


class CustomHelpFormatter(argparse.HelpFormatter):
    def _get_help_string(self, action):
        help_text = action.help
        if action.default is not argparse.SUPPRESS:
            help_text += f" (default: {action.default})"
        return help_text


class RollingFFTAverage:
    def __init__(self, freq: np.ndarray, window_size: int):
        """
        freq: example FFT vector (used for shape + dtype)
        window_size: number of frames in rolling average
        """
        self.size = window_size
        self.buffer = np.zeros((window_size, *freq.shape), dtype=freq.dtype)
        self.accum = np.zeros_like(freq)
        self.idx = 0
        self.count = 0

    def update(self, mag: np.ndarray) -> np.ndarray:
        mag2 = mag * mag  # ← do NOT modify input

        self.accum -= self.buffer[self.idx]
        self.buffer[self.idx] = mag2
        self.accum += mag2

        self.idx = (self.idx + 1) % self.size

        if self.count < self.size:
            self.count += 1

        return np.sqrt(self.accum / self.count)


def db_rel(k: float) -> float:
    return float("nan") if k <= 0 else 20.0 * math.log10(k)


def from_db(db: float) -> float:
    return 10.0 ** (db / 20.0)


def find_top_two_peaks(
    wm: np.ndarray,
    freqs: np.ndarray,
    fmask: np.ndarray,
    min_rel_level: float,
    min_sep_hz: float,
) -> list[int]:

    valid = np.where(fmask > 0.5)[0]
    if len(valid) < 3:
        return []

    mags = wm[valid]

    # strongest bins
    order = np.argsort(mags)[::-1]

    peaks = []

    for k in order:

        idx = valid[k]

        if wm[idx] <= 0:
            continue

        # check separation
        if any(abs(freqs[idx] - freqs[p]) < min_sep_hz for p in peaks):
            continue

        peaks.append(idx)

        if len(peaks) == 2:
            break

    if not peaks:
        return []

    # level test
    strongest_db = db_rel(wm[peaks[0]])

    filtered = []

    for p in peaks:

        level_db = db_rel(wm[p])

        if level_db >= strongest_db + db_rel(min_rel_level):
            filtered.append(p)

    return sorted(filtered, key=lambda i: freqs[i])


def clean_log(values: np.ndarray, floor: float = 1e-20) -> np.ndarray:
    clipped = np.maximum(values, floor)
    return 20.0 * np.log10(clipped)


class WRef:
    @staticmethod
    def normalize_unit(values: np.ndarray) -> np.ndarray:
        vmax = np.max(values)
        if vmax > 1e-6:
            return values / vmax
        return values.copy()

    @staticmethod
    def wclean(ts: np.ndarray, window: np.ndarray, center_freq: float) -> np.ndarray:
        clean = np.sin(2.0 * np.pi * center_freq * ts) * window
        spectrum = np.fft.rfft(clean)
        mag = WRef.normalize_unit(np.abs(spectrum) / len(ts))
        return mag

    def __init__(self, ts: np.ndarray, window: np.ndarray, freqs: np.ndarray):
        self.ref_idx = len(freqs) // 2
        ref_freq = freqs[self.ref_idx]
        self.ref = self.wclean(ts, window, ref_freq)
        self.rpeak = np.max(self.ref)

    def shift_kernel(self, target_idx: int, peak) -> np.ndarray:
        out = np.zeros_like(self.ref)
        delta = target_idx - self.ref_idx

        src_lo = max(0, -delta)
        src_hi = min(len(self.ref), len(self.ref) - delta)
        dst_lo = max(0, delta)
        dst_hi = min(len(self.ref), len(self.ref) + delta)

        if src_hi > src_lo and dst_hi > dst_lo:
            out[dst_lo:dst_hi] = self.ref[src_lo:src_hi]

        return out * peak / self.rpeak if self.rpeak > 1e-20 else out

def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))

def pratio1(a, b):
    return math.sqrt(max(a, EPS) / max(b, EPS))

def pratio2(a, b):
    k = pratio1(a, b)
    return db_rel(k), 100.0 * k
    
def thd(vfund, vdist):
    return pratio2(vdist, vfund)

def thdn(vfund, vdist, vnoise):
    return pratio2(vdist + vnoise, vfund)

def imd(vtone, vimd):
    return pratio2(vimd, vtone)

def sinad(vfund, vdist, vnoise):
    return db_rel(pratio1(vfund, vdist + vnoise))

def snr(vfund, vnoise):
    return db_rel(pratio1(vfund, vnoise))

def enob(snr_db):
    return (snr_db - 1.76) / 6.02

def nearest_index(arr: np.ndarray, value: float) -> int:
    return np.argmin(np.abs(arr - value))

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if (n % 2) == 0:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))


def nearest_fft_size(x: int) -> int:
    if x < 2:
        raise ValueError("FFT size must be >= 2")
    return 2 ** math.ceil(math.log2(x))


def noise_from_db(x: float) -> float:
    if x >= 0:
        return 1
    return 10 ** (x / 20.0)


def get_window(name: str, chunk: int) -> np.ndarray:
    key = name.strip().lower()
    if key == "bartlett":
        return np.bartlett(chunk)
    if key == "blackman":
        return np.blackman(chunk)
    if key == "hamming":
        return np.hamming(chunk)
    if key in {"hanning", "hann"}:
        return np.hanning(chunk)
    if key in {"rect", "rectangular", "none"}:
        return np.ones(chunk)
    raise ValueError(f"Unsupported window: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Yet Another Audio Analyzer",
        usage="%(prog)s [options]",
        formatter_class=CustomHelpFormatter,
    )

    parser.add_argument("--list", action="store_true")
    parser.add_argument("--freq", type=int, default=192000, help="Sample rate")
    parser.add_argument("--dev", type=int, default=4, help="ID of sound device")
    parser.add_argument(
        "--chsel", type=int, default=1, help="Selected channel (0-based)"
    )
    parser.add_argument("--chnum", type=int, default=2, help="Number of channels")
    parser.add_argument("--chunk", type=int, default=65536, help="FFT size")
    parser.add_argument("--adcrng", type=float, default=100.0, help="ADC voltage range")
    parser.add_argument(
        "--vrange", type=float, default=40.0, help="Display voltage range in V"
    )
    parser.add_argument(
        "--frange", type=float, default=70000.0, help="Displayed frequency range in Hz"
    )
    parser.add_argument(
        "--trange", type=float, default=100.0, help="Displayed time range in ms"
    )
    parser.add_argument(
        "--wrange", type=float, default=-150.0, help="FFT range lower bound in dB"
    )
    parser.add_argument("--rload", type=float, default=8.0, help="Load resistor in ohm")
    parser.add_argument(
        "--thd", type=int, default=7, help="Number of harmonics for THD"
    )
    parser.add_argument(
        "--duration", type=int, default=240, help="Time to exit in seconds"
    )
    parser.add_argument("--plot", type=str, default="", help="Write plot to file")
    parser.add_argument("--csv", type=str, default="", help="Append metrics to CSV")
    parser.add_argument("--window", type=str, default="hanning", help="Window function")
    parser.add_argument(
        "--simfreq", type=float, default=0.0, help="Simulate exact frequency in Hz"
    )
    parser.add_argument(
        "--simnoise", type=float, default=-160.0, help="Simulate noise amplitude in dB"
    )
    parser.add_argument(
        "--flttsh", type=float, default=120.0, help="Notch filter level in dB"
    )
    parser.add_argument(
        "--cftsh", type=float, default=0.25, help="Center frequency threshold in Hz"
    )
    parser.add_argument(
        "--twotone-rel-db",
        type=float,
        default=16.0,
        help="Second tone must be within this many dB of the main tone to enter IMD mode",
    )
    parser.add_argument(
        "--peak-sep-hz",
        type=float,
        default=20.0,
        help="Minimum separation between two fundamentals in Hz for IMD detection",
    )
    parser.add_argument(
        "--simfreq2", type=float, default=0.0, help="Second simulated tone frequency"
    )

    parser.add_argument(
        "--simamp2", type=float, default=1.0, help="Second simulated tone amplitude"
    )
    parser.add_argument(
        "--simhmncs", type=float, default=0.0, help="Add harmonics to the simulation"
    )
    return parser


def init_csv(path: str) -> None:
    if path and not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "Mode,"
                "F1_hz,F2_hz,"
                "THDN_dB,THDN_pct,"
                "THD_dB,THD_pct,"
                "CCIF_dB,CCIF_pct,"
                "SMPTE_dB,SMPTE_pct,"
                "SNR_dB,"
                "SINAD_db,ENOB,"
                "Vrms,Prms,"
                "FFT size,Sample Rate\n"
            )


def write_csv(
    csv,
    imd_mode,
    tone1_freq,
    tone2_freq,
    thdn_db,
    thdn_pct,
    thd_db,
    thd_pct,
    ccif_db,
    ccif_pct,
    smpte_db,
    smpte_pct,
    snr_db,
    sinad_db,
    enob_bits,
    vrms,
    prms,
    fft_size,
    sample_rate,
) -> None:
    if csv and os.path.isfile(csv):
        mode_name = "IMD" if imd_mode else "THD"
        with open(csv, "a", encoding="utf-8") as f:
            f.write(
                f"{mode_name},"
                f"{tone1_freq},{tone2_freq},"
                f"{thdn_db},{thdn_pct},"
                f"{thd_db},{thd_pct},"
                f"{ccif_db},{ccif_pct},"
                f"{smpte_db},{smpte_pct},"
                f"{snr_db},"
                f"{sinad_db},{enob_bits},"
                f"{vrms},{prms},"
                f"{fft_size},{sample_rate}\n"
            )


def fft_magnitude(meas: np.ndarray, window: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(meas * window)
    coherent_gain = np.sum(window) / len(window)
    enbw = np.sum(window**2) / (np.sum(window) ** 2) * len(window)
    mag = np.abs(spectrum) / (len(meas) * coherent_gain)
    mag /= math.sqrt(enbw)
    return mag


def compute_fft(meas: np.ndarray, window: np.ndarray, fmask: np.ndarray) -> np.ndarray:
    return fft_magnitude(meas, window) * fmask


def time_domain_analyse(meas, load_ohm):
    vpp = float(np.max(meas) - np.min(meas))
    ppeak = float(np.max(np.square(meas)) / load_ohm)
    vrms = rms(meas)
    prms = (vrms**2) / load_ohm
    return vpp, ppeak, vrms, prms


def prime_freq_list(freqs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = []
    for i in range(3, len(freqs)):
        if mask[i] > 0.5 and is_prime(i):
            out.append(freqs[i])
    return np.array(out, dtype=float)


def best_freq(prime_list: np.ndarray) -> np.ndarray:
    if len(prime_list) == 0:
        return np.array([], dtype=float)

    targets = [60.0, 250.0, 1000.0, 7000.0, 8000.0, 10000.0, 15000.0, 19000.0, 20000.0]
    indices = [int(np.argmin(np.abs(prime_list - t))) for t in targets]
    return prime_list[np.clip(indices, 0, len(prime_list) - 1)]


def get_fft_params(chunk, sample_rate, freq_range):
    freqs = np.fft.rfftfreq(chunk, d=1 / sample_rate)
    fmask = np.zeros(len(freqs), dtype=float)
    i_lo = nearest_index(freqs, freq_range[0])
    i_hi = nearest_index(freqs, freq_range[1])
    fmask[i_lo:i_hi+1] = 1.0
    prime_freqs = prime_freq_list(freqs, fmask)
    best_freqs = best_freq(prime_freqs)
    return freqs, fmask, i_lo, i_hi, best_freqs


def get_ts(chunk, sample_rate, trange):
    tmax = chunk / sample_rate
    time_range = [
        max((tmax - trange * 1e-3) * 0.5, 0.0),
        min((tmax + trange * 1e-3) * 0.5, tmax),
    ]
    return np.linspace(0.0, tmax, chunk, endpoint=False), time_range


def notch(values: np.ndarray, level: float) -> np.ndarray:
    mask = np.zeros_like(values, dtype=float)
    mask[values > level] = 1.0
    return mask


def build_single_tone_masks(
    mag: np.ndarray,
    freqs: np.ndarray,
    carrier_idx: int,
    wref: WRef,
    cfg: AudioConfig,
):
    carrier_freq = freqs[carrier_idx]
    wc = wref.shift_kernel(carrier_idx, mag[carrier_idx])
    tones_mask = notch(wc, cfg.flt_threshold)

    tmp = np.zeros_like(wc)
    for h in range(2, cfg.thd_harmonics + 1):
        harmonic_idx = carrier_idx * h
        if harmonic_idx >= len(wc):
            break

        w = wref.shift_kernel(harmonic_idx, mag[harmonic_idx])
        tmp += w

    hrmncs_mask = notch(tmp, cfg.flt_threshold)

    return wc, tones_mask, hrmncs_mask, carrier_freq

def build_two_tone_masks(
    mag: np.ndarray,
    freqs: np.ndarray,
    idx1: int,
    idx2: int,
    wref,
    cfg,
):
    freq1 = freqs[idx1]
    freq2 = freqs[idx2]

    wc1 = wref.shift_kernel(idx1, mag[idx1])
    wc2 = wref.shift_kernel(idx2, mag[idx2])

    # --- Combined tone mask ---
    wc = wc1 + wc2

    tone1_mask = notch(wc1, cfg.flt_threshold)
    tone2_mask = notch(wc2, cfg.flt_threshold)
    tones_mask = notch(wc, cfg.flt_threshold)

    # --- Intermodulation distortion indices ---
    imd_idx = [
        abs(idx1 - idx2),
        abs(2 * idx1 - idx2),
        abs(2 * idx2 - idx1),
        idx1 + idx2,
        idx1 * 2,
        idx1 * 3,
        idx1 * 4,
        idx2 * 2,
        idx2 * 3,
        idx2 * 4
    ]

    imd_mask = []
    tmp = np.zeros_like(wc)

    for idx in imd_idx:
       
        if 0 <= idx < len(wc):
            iw = wref.shift_kernel(idx, mag[idx])
            imask = notch(tmp, cfg.flt_threshold)
            tmp += iw
        else:
            imask = (np.zeros_like(wc))

        imd_mask.append(imask)

    harmonics_mask = notch(tmp, cfg.flt_threshold)

    return (
        wc,
        tones_mask,
        harmonics_mask,
        tone1_mask,
        tone2_mask,
        imd_mask,
        freq1,
        freq2,
    )


# --- Power helpers ---

def compute_power(mag: np.ndarray, mask: np.ndarray):
    return np.sum((mag * mask) ** 2)


def compute_powers(mag, tones_mask, harmonics_mask):
    vfund = compute_power(mag, tones_mask)
    vdist = compute_power(mag, harmonics_mask)

    noise_mask = 1.0 - np.clip(tones_mask + harmonics_mask, 0.0, 1.0)
    vnoise = compute_power(mag, noise_mask)

    vtotal = vfund + vdist + vnoise
    return vfund, vdist, vnoise, vtotal
##########################################

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    chunk = nearest_fft_size(args.chunk)

    cfg = AudioConfig(
        sample_rate=args.freq,
        chunk=chunk,
        device_index=args.dev if args.simfreq <= 0 else -1,
        channel_select=args.chsel,
        channel_count=args.chnum,
        adc_range=args.adcrng,
        load_ohm=args.rload,
        duration_s=args.duration,
        window_name=args.window,
        sim_freq=args.simfreq,
        sim_freq2=args.simfreq2,
        sim_noise=noise_from_db(args.simnoise),
        sim_amp2=args.simamp2,
        sim_hmncs=args.simhmncs,
        thd_harmonics=args.thd,
        flt_threshold=from_db(-args.flttsh),
        center_freq_threshold=args.cftsh,
        two_tone_rel_db=args.twotone_rel_db,
        peak_min_separation_hz=args.peak_sep_hz,
    )

    voltage_range = [-args.vrange, args.vrange]
    freq_range = [10.0, args.frange]
    db_range = [args.wrange, 0.0]

    stream = None

    try:
        if args.list:
            list_sound_devices()
            return 0

        window = get_window(cfg.window_name, cfg.chunk)
        init_csv(args.csv)

        pic_base, pic_ext = os.path.splitext(args.plot) if args.plot else ("", "")

        fig, (skip0, ax_time, skip1, ax_freq, skip2) = plt.subplots(
            5,
            1,
            figsize=(10, 6),
            gridspec_kw={"height_ratios": [0.05, 2.2, 0.15, 6.5, 2]},
        )

        fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.0, hspace=0.25)

        formatter_s = EngFormatter(unit="s")
        formatter_v = EngFormatter(unit="V")
        formatter_hz = EngFormatter(unit="Hz")
        formatter_db = EngFormatter(unit="dB")

        ring = RingBuffer((cfg.chunk), cfg.channel_count)
        stream = open_stream(cfg, ring)

        freqs, fmask, i_lo, i_hi, best_freqs = get_fft_params(
            cfg.chunk, cfg.sample_rate, freq_range
        )
        ts, time_range = get_ts(cfg.chunk, cfg.sample_rate, args.trange)
        wref = WRef(ts, window, freqs)
        fft_avg = RollingFFTAverage(freqs, 4)

        td_step = max(1, len(ts) // 4000)

        # ---- STATIC AXIS SETUP ----
        ax_time.set_title("Time Domain", loc="left")
        ax_time.set_xlim(time_range)
        ax_time.set_ylim(voltage_range)
        ax_time.xaxis.set_major_formatter(formatter_s)
        ax_time.yaxis.set_major_formatter(formatter_v)
        ax_time.grid()

        ax_freq.set_title("Frequency Domain", loc="left")
        ax_freq.set_xscale("log")
        ax_freq.set_xlim(freq_range)
        ax_freq.set_ylim(db_range)
        ax_freq.xaxis.set_major_formatter(formatter_hz)
        ax_freq.yaxis.set_major_formatter(formatter_db)
        ax_freq.grid()

        # ---- PLOT OBJECTS (CREATE ONCE) ----
        (time_line,) = ax_time.plot([], [], "r")
        (freq_line,) = ax_freq.plot([], [], "b-")
        (wc_line,) = ax_freq.plot([], [], "g.", markersize=3)

        # peak annotations storage
        peak_texts = []

        # ---- STATUS TEXT (CREATE ONCE) ----
        status_text1 = skip2.text(0.01, 0.60, "", fontfamily="monospace", fontsize=10)
        status_text2 = skip2.text(0.01, 0.40, "", fontfamily="monospace", fontsize=10)
        status_text3 = skip2.text(0.01, 0.20, "", fontfamily="monospace", fontsize=10)
        status_text4 = skip2.text(
            0.01, 0.00, "", fontfamily="monospace", fontsize=8, style="italic"
        )

        closed = {"value": False}

        def on_close(_event):
            closed["value"] = True

        fig.canvas.mpl_connect("close_event", on_close)
        fig.suptitle("Yet Another Audio analyzeR")
        skip0.axis("off")
        skip1.axis("off")
        skip2.axis("off")

        prev_carrier_idx = -1
        stable_count = 0
        write_after = 10

        t_start = time.time()
        while (time.time() - t_start < cfg.duration_s) and not closed["value"]:
            if stream is None:
                meas = simulate_signal(
                    cfg.sim_freq,
                    cfg.sim_noise,
                    ts,
                    cfg.sim_freq2,
                    cfg.sim_amp2,
                    cfg.sim_hmncs,
                    cfg.thd_harmonics,
                )
            else:
                meas = read_measurement(cfg, ring)
                if meas is None:
                    time.sleep(0.001)
                    continue

            if len(meas) < 16:
                raise RuntimeError("Measurement buffer too short.")

            # Time-domain metrics
            vpp, ppeak, vrms, prms = time_domain_analyse(meas, cfg.load_ohm)

            mag = fft_avg.update(compute_fft(meas, window, fmask))

            peak_indices = find_top_two_peaks(
                mag,
                freqs,
                fmask,
                min_rel_level=10 ** (-cfg.two_tone_rel_db / 20),
                min_sep_hz=cfg.peak_min_separation_hz,
            )
            imd_mode = len(peak_indices) >= 2
            carrier_idx = peak_indices[0] if peak_indices else int(np.argmax(mag))

            if imd_mode:
                idx1, idx2 = peak_indices[:2]
                (
                    wc,
                    tones_mask,
                    harmonics_mask,
                    tone1_mask,
                    tone2_mask,
                    imd_masks,
                    tone1_freq,
                    tone2_freq,
                ) = build_two_tone_masks(mag, freqs, 
                                        idx1, idx2, 
                                        wref, cfg)
                    
                vtone1 = compute_power(mag, tone1_mask)
                vtone2 = compute_power(mag, tone2_mask)
                vimd = [compute_power(mag, m) for m in imd_masks]
                vref = 0.5 * (vtone1 + vtone2)
                ccif_db, ccif_pct = imd(vref, vimd[0])
                smpte_db, smpte_pct = imd(vtone2, vimd[1] + vimd[2])
            else:
                (   
                    wc, 
                    tones_mask, 
                    harmonics_mask, 
                    tone1_freq
                ) = build_single_tone_masks(mag, freqs, 
                                        carrier_idx, 
                                        wref, cfg)
                tone2_freq = 0
                smpte_db = smpte_pct = ccif_db = ccif_pct = float("nan")

            vfund, vdist, vnoise, vtotal = compute_powers(mag, tones_mask, harmonics_mask)
            if not imd_mode:
                thd_db, thd_pct = thd(vfund, vdist)
                thdn_db, thdn_pct = thdn(vfund, vdist, vnoise)
            else:
                thd_db = thd_pct = float("nan")
                thdn_db = thdn_pct = float("nan")
            sinad_db = sinad(vfund, vdist, vnoise)
            snr_db = snr(vfund, vnoise)
            enob_bits = enob(snr_db)

            # ---- TIME PLOT (DECIMATED) ----
            time_line.set_data(ts[::td_step], meas[::td_step])

            # ---- FFT PLOT ----
            mag_db_full = clean_log(mag)
            wc_db_full = clean_log(wc)

            top = mag_db_full[i_lo:i_hi].max()

            freq_line.set_data(freqs[i_lo:i_hi], mag_db_full[i_lo:i_hi] - top)

            wc_line.set_data(freqs[i_lo:i_hi], wc_db_full[i_lo:i_hi] - top)

            # ---- PEAK ANNOTATIONS (reuse) ----
            for t in peak_texts:
                t.remove()
            peak_texts.clear()

            if not imd_mode and carrier_idx > 0:
                for i in range(1, 1 + cfg.thd_harmonics):
                    idx = carrier_idx * i
                    if idx >= len(mag):
                        break

                    y = mag[idx]
                    if y <= 1e-10:
                        continue

                    y_db = mag_db_full[idx] - mag_db_full.max()
                    if y_db <= db_range[0]:
                        continue

                    txt = ax_freq.text(
                        freqs[idx],
                        y_db,
                        str(i),
                        ha="center",
                        va="bottom",
                        color="c",
                        fontstyle="italic",
                    )
                    peak_texts.append(txt)

            else:
                for tone_freq, label in [
                    (tone1_freq, "F1"),
                    (tone2_freq, "F2"),
                ]:
                    if tone_freq <= 0:
                        continue

                    idx = nearest_index(freqs, tone_freq)
                    y = mag[idx]
                    if y <= 1e-10:
                        continue

                    y_db = mag_db_full[idx] - mag_db_full.max()
                    if y_db <= db_range[0]:
                        continue

                    txt = ax_freq.text(
                        freqs[idx],
                        y_db,
                        label,
                        ha="center",
                        va="bottom",
                        color="c",
                        fontstyle="italic",
                    )
                    peak_texts.append(txt)

            # ---- STATUS TEXT UPDATE ----
            if imd_mode:
                line1 = (
                    f"{'F1':<6}{tone1_freq:>8.2f} Hz   "
                    f"{'F2':<6}{tone2_freq:>8.2f} Hz   "
                    f"{'CCIF':<6}{ccif_db:>7.2f} dB ({ccif_pct:6.2f} %)   "
                    f"{'SMPTE':<6}{smpte_db:>7.2f} dB ({smpte_pct:6.3f} %)"
                )
            else:
                line1 = (
                    f"{'BASE':<6}{tone1_freq:>8.2f} Hz   "
                    f"{'':<16}    "
                    f"{'THD':<6}{thd_db:>7.2f} dB ({thd_pct:6.2f} %)   "
                    f"{'THD+N':<6}{thdn_db:>7.2f} dB ({thdn_pct:6.2f} %)"
                )

            line2 = (
                f"{'FFT':<6}{cfg.chunk:>8d}      "
                f"{'SR':<6}{cfg.sample_rate/1000:>8.1f} kHz  "
                f"{'SNR':<6}{snr_db:>7.2f} dB              "
                f"{'ENOB':<6}{enob_bits:>7.2f} bits"
            )

            line3 = (
                f"{'Vpp':<6}{vpp:>8.2f} V    "
                f"{'Vrms':<6}{vrms:>8.2f} V    "
                f"{'Prms':<6}{prms:>7.2f} W               "
                f"{'LOAD':<6}{cfg.load_ohm:>7.1f} Ω"
            )

            ref_line = f"{'REF':<6}"
            for bf in best_freqs:
                mark = "*" if abs(tone1_freq - bf) < 1e-6 else " "
                ref_line += f"{bf:>8.2f} Hz{mark} "

            status_text1.set_text(line1)
            status_text2.set_text(line2)
            status_text3.set_text(line3)
            status_text4.set_text(ref_line)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            if carrier_idx == prev_carrier_idx:
                stable_count += 1
            else:
                prev_carrier_idx = carrier_idx
                stable_count = 0

            if stable_count == write_after:
                if pic_base:
                    plt.savefig(
                        f"{pic_base}_{tone1_freq:.0f}Hz_{tone2_freq:.0f}Hz{pic_ext}"
                    )

                write_csv(
                    args.csv,
                    imd_mode,
                    tone1_freq,
                    tone2_freq,
                    thdn_db,
                    thdn_pct,
                    thd_db,
                    thd_pct,
                    ccif_db,
                    ccif_pct,
                    smpte_db,
                    smpte_pct,
                    snr_db,
                    sinad_db,
                    enob_bits,
                    vrms,
                    prms,
                    cfg.chunk,
                    cfg.sample_rate,
                )

        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            if stream is not None:
                stream.stop()
                stream.close()
        finally:
            plt.close("all")


if __name__ == "__main__":
    raise SystemExit(main())
