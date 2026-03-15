#! /usr/bin/env python3
#
# Yet Another Audio analyzeR
#
# Copyright 2024 George Biro
#
# GPLv3-or-later
#

from __future__ import annotations

import ctypes
from ctypes import c_uint32, c_void_p, byref
import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from matplotlib.ticker import EngFormatter

plt.ion()
EPS = 1e-20


class CustomHelpFormatter(argparse.HelpFormatter):
    def _get_help_string(self, action):
        help_text = action.help
        if action.default is not argparse.SUPPRESS:
            help_text += f" (default: {action.default})"
        return help_text


@dataclass
class AudioConfig:
    sample_rate: int
    chunk: int
    skip: int
    device_index: int
    channel_select: int
    channel_count: int
    adc_range: float
    adc_resolution: int
    load_ohm: float
    duration_s: int
    window_name: str
    avg_enabled: bool
    sim_freq: float
    sim_freq2: float
    sim_noise: float
    thd_harmonics: int
    flt_threshold: float
    fnd_threshold: float
    frq_threshold: float
    center_freq_threshold: float
    two_tone_rel_db: float
    peak_min_separation_hz: float

@dataclass
class AdcFormat:
    pa_format: int
    dtype: np.dtype
    scale: float

@dataclass
class ToneState:
    imd_mode: bool
    carrier_idx: int
    tone1_freq: float
    tone2_freq: float
    tone1_mask: np.ndarray
    tone2_mask: np.ndarray
    fundamental_mask: np.ndarray
    harmonics_mask: np.ndarray
    analysis_filter: np.ndarray
    wc: np.ndarray

def list_sound_devices(audio: pyaudio.PyAudio) -> None:
    host = 0
    info = audio.get_host_api_info_by_index(host)
    num_devices = info.get("deviceCount", 0)
    for i in range(num_devices):
        dev = audio.get_device_info_by_host_api_device_index(host, i)
        if dev.get("maxInputChannels", 0) > 0:
            print(f"Input Device id {i} - {dev.get('name')}")

def db_rel(k: float) -> float:
    return float("nan") if k <= 0 else 20.0 * math.log10(k)

def db_pow(k: float) -> float:
    return float("nan") if k <= 0 else 10.0 * math.log10(k)

def from_db(db: float) -> float:
    return 10.0 ** (db / 20.0)

def notch(values: np.ndarray, level: float) -> np.ndarray:
    mask = np.zeros_like(values, dtype=float)
    mask[values > level] = 1.0
    return mask

def carrier(
    spectrum: np.ndarray, num_harmonics: int, freq_mask: np.ndarray, level: float
) -> Tuple[int, np.ndarray, np.ndarray]:
    carrier_idx = int(np.argmax(spectrum))

    if carrier_idx < 1:
        step = 1
        harmonic_count = 1
    else:
        step = carrier_idx
        harmonic_count = num_harmonics

    fundamental_mask = notch(spectrum, spectrum[carrier_idx] * level)

    harmonics_mask = np.zeros_like(spectrum, dtype=float)
    harmonics_mask[carrier_idx::step][:harmonic_count] = 1.0
    harmonics_mask[carrier_idx] = 0.0
    harmonics_mask *= freq_mask

    return carrier_idx, fundamental_mask, harmonics_mask

def build_peak_mask(freqs: np.ndarray, center_freq: float, half_width_hz: float) -> np.ndarray:
    mask = np.zeros(len(freqs), dtype=float)
    if center_freq <= 0:
        return mask
    lo = center_freq - half_width_hz
    hi = center_freq + half_width_hz
    mask[(freqs >= lo) & (freqs <= hi)] = 1.0
    return mask

def peak_band_from_synth(
    ts: np.ndarray,
    window: np.ndarray,
    freqs: np.ndarray,
    center_freq: float,
    floor_level: float,
) -> np.ndarray:
    wc = wclean(ts, window, center_freq, floor_level)
    return notch(wc, 1e-20)

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
    strongest = wm[peaks[0]]

    peaks = [p for p in peaks if wm[p] >= strongest * min_rel_level]

    return sorted(peaks, key=lambda i: freqs[i])

def calc_peak_freq(wm, freqs, idx, _rel):

    if idx <= 0 or idx >= len(wm) - 1:
        return freqs[idx]

    a = wm[idx - 1]
    b = wm[idx]
    c = wm[idx + 1]

    denom = a - 2 * b + c
    if abs(denom) < 1e-20:
        return freqs[idx]

    delta = 0.5 * (a - c) / denom

    bin_width = freqs[1] - freqs[0]

    return freqs[idx] + delta * bin_width

def imd_total(
    wm: np.ndarray,
    mfund1: np.ndarray,
    mfund2: np.ndarray,
    fmask: np.ndarray,
) -> tuple[float, float]:
    """
    General-purpose IMD metric:
    unwanted energy divided by wanted energy, where wanted energy
    is the sum of the two fundamentals.
    """
    msig = np.clip(mfund1 + mfund2, 0.0, 1.0)
    mnoise = np.clip(fmask - msig, 0.0, 1.0)

    vsig = np.sum(np.square(wm * msig))
    vdist = np.sum(np.square(wm * mnoise))

    if vsig < 1e-100:
        return float("nan"), float("nan")

    k = math.sqrt(vdist / vsig)
    return db_rel(k), 100.0 * k

def imd_ccif_difference(
    wm: np.ndarray,
    freqs: np.ndarray,
    f1: float,
    f2: float,
    half_width_hz: float,
) -> tuple[float, float, float]:
    """
    For CCIF-like two-tone tests, report the difference product |f2-f1|.
    Returns: product frequency, level relative to RMS of the two tones, percent
    """
    fd = abs(f2 - f1)
    if fd <= 0:
        return float("nan"), float("nan"), float("nan")

    diff_mask = build_peak_mask(freqs, fd, half_width_hz)
    vdiff = np.sum(np.square(wm * diff_mask))

    m1 = build_peak_mask(freqs, f1, half_width_hz)
    m2 = build_peak_mask(freqs, f2, half_width_hz)
    vref = np.sum(np.square(wm * (m1 + m2)))

    if vref < 1e-100:
        return fd, float("nan"), float("nan")

    k = math.sqrt(vdiff / vref)
    return fd, db_rel(k), 100.0 * k

def calc_f_freq(spectrum: np.ndarray, freqs: np.ndarray, level: float) -> float:
    weighted = notch(spectrum, level) * spectrum
    denom = np.sum(weighted)
    if denom <= EPS:
        return 0.0
    return float(np.sum(weighted * freqs) / denom)


def normalize_unit(values: np.ndarray) -> np.ndarray:
    vmax = np.max(values)
    if vmax > 1e-6:
        return values / vmax
    return values.copy()


def clean_log(values: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    clipped = np.maximum(values, floor)
    return 20.0 * np.log10(clipped)


def wclean(ts: np.ndarray, window: np.ndarray, center_freq: float, floor_level: float) -> np.ndarray:
    clean = np.sin(2.0 * np.pi * center_freq * ts) * window
    spectrum = np.fft.rfft(clean)
    mag = normalize_unit(np.abs(spectrum) / len(spectrum))
    mag[mag < floor_level] = 0.0
    return mag

def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))

def thd_ieee(wm, mh, carrier_idx):
    vfund = np.sum(np.square(wm * build_peak_mask(
        np.arange(len(wm)), carrier_idx, 1)))
    vharm = np.sum(np.square(wm * mh))
    if vfund < 1e-100:
        return float("nan"), float("nan")

    k = math.sqrt(vharm / vfund)
    return db_rel(k), 100*k

def thdn(wm: np.ndarray, mfund: np.ndarray, mfilter: np.ndarray, fmask: np.ndarray) -> Tuple[float, float]:
    vfund = np.sum(np.square(wm * mfund))
    noise_mask = fmask - mfilter
    vnoise = np.sum(np.square(wm * noise_mask))
    if vfund < 1e-100:
        return float("nan"), float("nan")
    k = math.sqrt(vnoise / vfund)
    return db_rel(k), 100.0 * k


def snr(wm: np.ndarray, mfund: np.ndarray, mfilter: np.ndarray, fmask: np.ndarray, mh: np.ndarray) -> float:
    vsignal = np.sum(np.square(wm * mfund))
    noise_mask = fmask - mfilter - mh
    vnoise = np.sum(np.square(wm * noise_mask))
    if vnoise < 1e-100:
        return float("nan")
    k = math.sqrt(vsignal / vnoise)
    return db_rel(k)


def enob(sinad_db: float) -> float:
    return (sinad_db - 1.76) / 6.02


def nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(arr - abs(value))))


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if (n % 2) == 0:
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))



def get_adc_format(bits: int) -> AdcFormat:
    if bits == 16:
        return AdcFormat(pyaudio.paInt16, np.int16, 2**15)
    if bits == 24:
        # Stored in 32-bit container for PyAudio compatibility.
        return AdcFormat(pyaudio.paInt32, np.int32, 2**24)
    if bits == 32:
        return AdcFormat(pyaudio.paFloat32, np.float32, 1.0)
    raise ValueError("Invalid ADC resolution. Use 16, 24, or 32.")


def nearest_fft_size(x: int) -> int:
    if x < 2:
        raise ValueError("FFT size must be >= 2")
    return 2 ** math.floor(math.log2(x) + 0.5)


def noise_from_db(x: float) -> float:
    if x < 0:
        return 0.0
    return 10 ** (-x / 20.0)


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


def simulate_signal(
    freq_hz: float,
    noise_amp: float,
    ts: np.ndarray,
    window: np.ndarray,
    freq2_hz: float = 0.0,
    amp2: float = 1.0,
) -> np.ndarray:
    signal = np.sin(2.0 * np.pi * freq_hz * ts + random.random() * np.pi)
    if freq2_hz > 0.0:
        signal += amp2 * np.sin(2.0 * np.pi * freq2_hz * ts + random.random() * np.pi)
    if noise_amp > 1e-6:
        signal += np.random.normal(0.0, noise_amp, len(signal))
    return signal * window

def get_coreaudio_device_id(audio: pyaudio.PyAudio, pa_index: int) -> int:
    info = audio.get_device_info_by_index(pa_index)
    return int(info["index"])

def open_stream(audio: pyaudio.PyAudio, cfg: AudioConfig, adc: AdcFormat) -> Optional[pyaudio.Stream]:
    if cfg.device_index < 0 or cfg.sim_freq >= 0.01:
        return None

    device_info = audio.get_device_info_by_index(cfg.device_index)
    max_channels = int(device_info.get("maxInputChannels", 0))
    if max_channels < 1:
        raise RuntimeError(f"Device '{device_info.get('name')}' has no input channels.")

    if cfg.channel_count > max_channels:
        print(
            f"Warning: Device '{device_info.get('name')}' supports only "
            f"{max_channels} input channel(s). Adjusting from {cfg.channel_count}."
        )
        cfg.channel_count = max_channels

    if cfg.channel_select < 0 or cfg.channel_select >= cfg.channel_count:
        raise ValueError(
            f"--chsel must be between 0 and {cfg.channel_count - 1} for --chnum {cfg.channel_count}"
        )

    return audio.open(
        format=adc.pa_format,
        rate=cfg.sample_rate,
        channels=cfg.channel_count,
        input=True,
        input_device_index=cfg.device_index,
        frames_per_buffer=cfg.chunk + cfg.skip,
        start=True,
    )


def read_measurement(stream: pyaudio.Stream, cfg: AudioConfig, adc: AdcFormat, window: np.ndarray) -> np.ndarray:
    data = stream.read(cfg.chunk + cfg.skip, exception_on_overflow=False)
    meas_full = np.frombuffer(data, dtype=adc.dtype)

    meas_full = meas_full[cfg.skip * cfg.channel_count :]
    meas_raw = meas_full[cfg.channel_select :: cfg.channel_count]
    return meas_raw[: cfg.chunk] * window * (cfg.adc_range / adc.scale)

def disable_safety_offset(device_id: int):

    coreaudio = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
    )

    UInt32 = ctypes.c_uint32

    class AudioObjectPropertyAddress(ctypes.Structure):
        _fields_ = [
            ("mSelector", UInt32),
            ("mScope", UInt32),
            ("mElement", UInt32),
        ]

    kAudioDevicePropertySafetyOffset = 1935764583
    kAudioObjectPropertyScopeInput = 1768845428
    kAudioObjectPropertyElementMaster = 0

    addr = AudioObjectPropertyAddress(
        kAudioDevicePropertySafetyOffset,
        kAudioObjectPropertyScopeInput,
        kAudioObjectPropertyElementMaster,
    )

    value = UInt32(0)
    size = UInt32(ctypes.sizeof(value))

    coreaudio.AudioObjectSetPropertyData(
        UInt32(device_id),
        ctypes.byref(addr),
        0,
        None,
        size,
        ctypes.byref(value),
    )

    print("Safety offset disabled")

def disable_voice_processing(device_index: int):
    """
    Disable CoreAudio voice processing (AGC, echo cancel, HPF)
    for the selected input device.
    """

    try:
        coreaudio = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
        )

        UInt32 = ctypes.c_uint32

        class AudioObjectPropertyAddress(ctypes.Structure):
            _fields_ = [
                ("mSelector", UInt32),
                ("mScope", UInt32),
                ("mElement", UInt32),
            ]

        # CoreAudio constants
        kAudioDevicePropertyVoiceProcessingEnable = 1987078511
        kAudioObjectPropertyScopeInput = 1768845428
        kAudioObjectPropertyElementMaster = 0

        addr = AudioObjectPropertyAddress(
            kAudioDevicePropertyVoiceProcessingEnable,
            kAudioObjectPropertyScopeInput,
            kAudioObjectPropertyElementMaster,
        )

        value = UInt32(0)
        size = UInt32(ctypes.sizeof(value))

        device_id = UInt32(device_index)

        coreaudio.AudioObjectSetPropertyData(
            device_id,
            ctypes.byref(addr),
            0,
            None,
            size,
            ctypes.byref(value),
        )

        print(f"Voice processing disabled for device {device_index}")

    except Exception as e:
        print("Voice processing disable failed:", e)

def enable_pro_audio_mode(device_id: int):
    """
    Enable CoreAudio pro-audio behavior:
    - hog mode (exclusive access)
    - minimal safety offset
    - disables most system DSP
    """

    try:
        coreaudio = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
        )

        UInt32 = ctypes.c_uint32
        pid = UInt32(os.getpid())

        class AudioObjectPropertyAddress(ctypes.Structure):
            _fields_ = [
                ("mSelector", UInt32),
                ("mScope", UInt32),
                ("mElement", UInt32),
            ]

        # constants
        kAudioDevicePropertyHogMode = 1752132965
        kAudioObjectPropertyScopeGlobal = 1735159650
        kAudioObjectPropertyElementMaster = 0

        addr = AudioObjectPropertyAddress(
            kAudioDevicePropertyHogMode,
            kAudioObjectPropertyScopeGlobal,
            kAudioObjectPropertyElementMaster,
        )

        size = UInt32(ctypes.sizeof(pid))

        coreaudio.AudioObjectSetPropertyData(
            UInt32(device_id),
            ctypes.byref(addr),
            0,
            None,
            size,
            ctypes.byref(pid),
        )

        print(f"Pro Audio (hog) mode enabled for device {device_id}")

    except Exception as e:
        print("Could not enable Pro Audio mode:", e)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Yet Another Audio Analyzer",
        usage="%(prog)s [options]",
        formatter_class=CustomHelpFormatter,
    )

    parser.add_argument("--list", action="store_true")
    parser.add_argument("--freq", type=int, default=192000, help="Sample rate")
    parser.add_argument("--dev", type=int, default=4, help="ID of sound device")
    parser.add_argument("--chsel", type=int, default=0, help="Selected channel (0-based)")
    parser.add_argument("--chnum", type=int, default=2, help="Number of channels")
    parser.add_argument("--chunk", type=int, default=32768, help="FFT size")
    parser.add_argument("--skip", type=int, default=1024, help="Samples to skip")
    parser.add_argument("--adcrng", type=float, default=100.0, help="ADC voltage range")
    parser.add_argument("--adcres", type=int, default=32, help="ADC resolution")
    parser.add_argument("--vrange", type=float, default=40.0, help="Display voltage range in V")
    parser.add_argument("--frange", type=float, default=70000.0, help="Displayed frequency range in Hz")
    parser.add_argument("--trange", type=float, default=100.0, help="Displayed time range in ms")
    parser.add_argument("--wrange", type=float, default=-150.0, help="FFT range lower bound in dB")
    parser.add_argument("--rload", type=float, default=8.0, help="Load resistor in ohm")
    parser.add_argument("--thd", type=int, default=7, help="Number of harmonics for THD")
    parser.add_argument("--duration", type=int, default=240, help="Time to exit in seconds")
    parser.add_argument("--plot", type=str, default="", help="Write plot to file")
    parser.add_argument("--csv", type=str, default="", help="Append metrics to CSV")
    parser.add_argument("--window", type=str, default="hanning", help="Window function")
    parser.add_argument("--avg", action="store_true", help="Enable averaging")
    parser.add_argument("--simfreq", type=float, default=0.0, help="Simulate exact frequency in Hz")
    parser.add_argument("--simnoise", type=float, default=-1.0, help="Simulate noise amplitude in dB")
    parser.add_argument("--flttsh", type=float, default=120.0, help="Notch filter level in dB")
    parser.add_argument("--fndtsh", type=float, default=3.0, help="Fundamental voltage threshold in dB")
    parser.add_argument("--frqtsh", type=float, default=40.0, help="Fundamental frequency threshold in dB")
    parser.add_argument("--cftsh", type=float, default=0.25, help="Center frequency threshold in Hz")
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
    parser.add_argument("--simfreq2", type=float, default=0.0,
                    help="Second simulated tone frequency")

    return parser


def init_csv(path: str) -> None:
    if path and not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "Mode,F1,F2,THD_dB,THD_pct,IMD_dB,IMD_pct,CCIF_diff_Hz,CCIF_diff_dB,CCIF_diff_pct,"
                "THDN_dB,THDN_pct,SNR_dB,ENOB,Vrms,Prms\n"
            )

def compute_fft(meas, window, fmask):
    spectrum_complex = np.fft.rfft(meas) * fmask
    coherent_gain = np.sum(window) / len(window)
    mag = np.abs(spectrum_complex) / (len(spectrum_complex) * coherent_gain)
    return normalize_unit(mag)

def plot_time(ax_time, ts, meas, time_range, voltage_range, formatter_s, formatter_v):
    ax_time.cla()
    ax_time.set_title("Time Domain", loc="left")
    ax_time.set_xlim(time_range)
    ax_time.set_ylim(voltage_range)
    ax_time.xaxis.set_major_formatter(formatter_s)
    ax_time.yaxis.set_major_formatter(formatter_v)
    ax_time.plot(ts, meas, "r")
    ax_time.grid()

def plot_freq(ax_freq, freqs, mag, wc, i_lo, i_hi,
              freq_range, db_range, formatter_hz, formatter_db):

    ax_freq.cla()
    ax_freq.set_title("Frequency Domain", loc="left")
    ax_freq.set_xscale("log")
    ax_freq.set_xlim(freq_range)
    ax_freq.set_ylim(db_range)

    ax_freq.xaxis.set_major_formatter(formatter_hz)
    ax_freq.yaxis.set_major_formatter(formatter_db)

    ax_freq.plot(freqs[i_lo:i_hi], clean_log(mag[i_lo:i_hi]), "b-")
    ax_freq.plot(freqs[i_lo:i_hi], clean_log(wc[i_lo:i_hi]), "g.")
    ax_freq.grid()

def render_status(skip2, imd_mode,
                  tone1_freq, tone2_freq,
                  thd_db, thd_pct,
                  imd_db, imd_pct,
                  imd_diff_db, imd_diff_pct,
                  sinad_db, sinad_pct,
                  snr_db, enob_bits,
                  vpp, vrms, prms,
                  cfg, best_freqs):

    skip2.cla()
    skip2.axis("off")

    if imd_mode:
        line1 = (
            f"{'F1':<6}{tone1_freq:>8.2f} Hz   "
            f"{'F2':<6}{tone2_freq:>8.2f} Hz   "
            f"{'IMD':<6}{imd_db:>7.2f} dB ({imd_pct:6.2f} %)   "
            f"{'CCIF':<6}{imd_diff_db:>7.2f} dB ({imd_diff_pct:6.3f} %)"
        )
    else:
        line1 = (
            f"{'BASE':<6}{tone1_freq:>8.2f} Hz   "
            f"{'':<16}    "
            f"{'THD':<6}{thd_db:>7.2f} dB ({thd_pct:6.2f} %)   "
            f"{'THD+N':<6}{sinad_db:>7.2f} dB ({sinad_pct:6.2f} %)"
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
        ref_line += f"{bf:>10.2f} Hz{mark}  "

    skip2.text(0.01,0.60,line1,fontfamily="monospace",fontsize=10)
    skip2.text(0.01,0.40,line2,fontfamily="monospace",fontsize=10)
    skip2.text(0.01,0.20,line3,fontfamily="monospace",fontsize=10)
    skip2.text(0.01,0.00,ref_line,fontfamily="monospace",fontsize=8,style="italic")

def time_domain_analyse(meas, load_ohm):
    vpp = float(np.max(meas) - np.min(meas))
    ppeak = float(np.max(np.square(meas)) / load_ohm)
    vrms = rms(meas)
    prms = (vrms ** 2) / load_ohm
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

    targets = [60.0, 1000.0, 7000.0, 10000.0, 19000.0, 20000.0]
    indices = [int(np.argmin(np.abs(prime_list - t))) for t in targets]
    return prime_list[np.clip(indices, 0, len(prime_list) - 1)]

def get_fft_params(chunk, sample_rate, freq_range):
    freqs = np.abs(np.fft.rfftfreq(chunk) * sample_rate)
    fmask = np.zeros(len(freqs), dtype=float)
    i_lo = nearest_index(freqs, freq_range[0])
    i_hi = nearest_index(freqs, freq_range[1])
    fmask[i_lo:i_hi] = 1.0
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

def analyze_tones(mag, freqs, ts, window, fmask, cfg) -> ToneState:

    peak_indices = find_top_two_peaks(
        mag,
        freqs,
        fmask,
        min_rel_level=10 ** (-cfg.two_tone_rel_db / 20),
        min_sep_hz=cfg.peak_min_separation_hz,
    )

    imd_mode = len(peak_indices) >= 2

    carrier_idx = peak_indices[0] if peak_indices else int(np.argmax(mag))
    carrier_freq = freqs[carrier_idx]

    wc = np.zeros_like(mag)

    if not imd_mode:

        fundamental_mask = notch(mag, mag[carrier_idx] * cfg.fnd_threshold)

        harmonics_mask = np.zeros_like(mag)

        if carrier_idx > 0:
            harmonics_mask[carrier_idx::carrier_idx][: cfg.thd_harmonics] = 1.0
            harmonics_mask[carrier_idx] = 0.0
            harmonics_mask *= fmask

        fundamental_freq = calc_peak_freq(
            mag, freqs, carrier_idx, cfg.frq_threshold
        )

        chosen_freq = (
            carrier_freq
            if abs(fundamental_freq - carrier_freq) < cfg.center_freq_threshold
            else fundamental_freq
        )

        wc = wclean(ts, window, chosen_freq, cfg.flt_threshold)

        analysis_filter = notch(wc, 1e-20) * fmask

        tone1_freq = chosen_freq
        tone2_freq = 0.0

        tone1_mask = fundamental_mask
        tone2_mask = np.zeros_like(mag)

    else:

        idx1, idx2 = peak_indices[:2]

        tone1_freq = calc_peak_freq(mag, freqs, idx1, cfg.frq_threshold)
        tone2_freq = calc_peak_freq(mag, freqs, idx2, cfg.frq_threshold)

        tone1_mask = peak_band_from_synth(
            ts, window, freqs, tone1_freq, cfg.flt_threshold
        ) * fmask

        tone2_mask = peak_band_from_synth(
            ts, window, freqs, tone2_freq, cfg.flt_threshold
        ) * fmask

        fundamental_mask = np.clip(tone1_mask + tone2_mask, 0.0, 1.0)

        harmonics_mask = np.zeros_like(mag)

        analysis_filter = fundamental_mask

        wc = tone1_mask + tone2_mask

    return ToneState(
        imd_mode=imd_mode,
        carrier_idx=carrier_idx,
        tone1_freq=tone1_freq,
        tone2_freq=tone2_freq,
        tone1_mask=tone1_mask,
        tone2_mask=tone2_mask,
        fundamental_mask=fundamental_mask,
        harmonics_mask=harmonics_mask,
        analysis_filter=analysis_filter,
        wc=wc,
    )

def compute_metrics(mag, tones, freqs, fmask, cfg):

    if not tones.imd_mode:

        thd_db, thd_pct = thd_ieee(
            mag,
            tones.harmonics_mask,
            tones.carrier_idx,
        )

        sinad_db, sinad_pct = thdn(
            mag,
            tones.fundamental_mask,
            tones.analysis_filter,
            fmask,
        )

        signal = np.sum(np.square(mag * tones.fundamental_mask))
        noise = np.sum(np.square(mag * (fmask - tones.fundamental_mask)))

        snr_db = db_rel(math.sqrt(signal / noise))

        enob_bits = enob(sinad_db)

        imd_db = float("nan")
        imd_pct = float("nan")
        imd_diff_freq = float("nan")
        imd_diff_db = float("nan")
        imd_diff_pct = float("nan")

    else:

        imd_db, imd_pct = imd_total(
            mag,
            tones.tone1_mask,
            tones.tone2_mask,
            fmask,
        )

        sinad_db, sinad_pct = thdn(
            mag,
            tones.fundamental_mask,
            tones.analysis_filter,
            fmask,
        )

        snr_db = snr(
            mag,
            tones.fundamental_mask,
            tones.analysis_filter,
            fmask,
            np.zeros_like(mag),
        )

        enob_bits = enob(sinad_db)

        thd_db = float("nan")
        thd_pct = float("nan")

        bin_width_hz = cfg.sample_rate / cfg.chunk

        imd_diff_freq, imd_diff_db, imd_diff_pct = imd_ccif_difference(
            mag,
            freqs,
            tones.tone1_freq,
            tones.tone2_freq,
            half_width_hz=max(2.0 * bin_width_hz, 2.0),
        )

    return (
        thd_db,
        thd_pct,
        imd_db,
        imd_pct,
        imd_diff_freq,
        imd_diff_db,
        imd_diff_pct,
        sinad_db,
        sinad_pct,
        snr_db,
        enob_bits,
    )

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    chunk = nearest_fft_size(args.chunk)
    adc = get_adc_format(args.adcres)

    cfg = AudioConfig(
        sample_rate=args.freq,
        chunk=chunk,
        skip=args.skip,
        device_index=args.dev,
        channel_select=args.chsel,
        channel_count=args.chnum,
        adc_range=args.adcrng,
        adc_resolution=args.adcres,
        load_ohm=args.rload,
        duration_s=args.duration,
        window_name=args.window,
        avg_enabled=args.avg,
        sim_freq=args.simfreq,
        sim_freq2=args.simfreq2,
        sim_noise=noise_from_db(args.simnoise),
        thd_harmonics=args.thd,
        flt_threshold=from_db(-args.flttsh),
        fnd_threshold=from_db(-args.fndtsh),
        frq_threshold=from_db(-args.frqtsh),
        center_freq_threshold=args.cftsh,
        two_tone_rel_db=args.twotone_rel_db,
        peak_min_separation_hz=args.peak_sep_hz,
    )

    voltage_range = [-args.vrange, args.vrange]
    freq_range = [10.0, args.frange]
    db_range = [args.wrange, 0.0]

    audio = pyaudio.PyAudio()
    stream = None
    
    try:
        if args.list:
            list_sound_devices(audio)
            return 0

        device_id = get_coreaudio_device_id(audio, cfg.device_index)
        disable_voice_processing(device_id)
        disable_safety_offset(device_id)
        enable_pro_audio_mode(device_id)

        window = get_window(cfg.window_name, cfg.chunk)
        init_csv(args.csv)

        pic_base, pic_ext = os.path.splitext(args.plot) if args.plot else ("", "")

        fig, (skip0, ax_time, skip1, ax_freq, skip2) = plt.subplots(
            5, 1, figsize=(10, 6),
            gridspec_kw={"height_ratios": [0.05, 2.2, 0.15, 6.5, 2]}
        )

        fig.subplots_adjust(
            left=0.07,
            right=0.98,
            top=0.93,
            bottom=0.0,
            hspace=0.25
        )

        closed = {"value": False}

        def on_close(_event):
            closed["value"] = True

        fig.canvas.mpl_connect("close_event", on_close)
        fig.suptitle("Yet Another Audio analyzeR")
        skip0.axis("off")
        skip1.axis("off")
        skip2.cla()
        skip2.axis("off")

        formatter_s = EngFormatter(unit="s")
        formatter_v = EngFormatter(unit="V")
        formatter_hz = EngFormatter(unit="Hz")
        formatter_db = EngFormatter(unit="dB")

        freqs, fmask, i_lo, i_hi, best_freqs = get_fft_params(cfg.chunk, cfg.sample_rate, freq_range)

        ts, time_range = get_ts(cfg.chunk, cfg.sample_rate, args.trange)

        prev_carrier_idx = -1
        stable_count = 0
        write_after = 10 if cfg.avg_enabled else 3

        stream = open_stream(audio, cfg, adc)
        
        t_start = time.time()
        while (time.time() - t_start < cfg.duration_s) and not closed["value"]:
            if stream is None:
                meas = simulate_signal(cfg.sim_freq, cfg.sim_noise, ts, window, cfg.sim_freq2)
            else:
                meas = read_measurement(stream, cfg, adc, window)

            if len(meas) < 16:
                raise RuntimeError("Measurement buffer too short.")

            # Time-domain plot
            plot_time(ax_time, ts, meas, time_range, voltage_range, formatter_s, formatter_v)

            # Time-domain metrics
            vpp, ppeak, vrms, prms = time_domain_analyse(meas, cfg.load_ohm)

            # FFT
            mag = compute_fft(meas, window, fmask)

            if len(mag) != len(freqs):
                raise RuntimeError("Spectrum length mismatch.")

            tones = analyze_tones(mag, freqs, ts, window, fmask, cfg)

            tone1_freq = tones.tone1_freq
            tone2_freq = tones.tone2_freq
            carrier_idx = tones.carrier_idx
            wc = tones.wc
            imd_mode = tones.imd_mode
            fundamental_mask = tones.fundamental_mask
            harmonics_mask = tones.harmonics_mask
            tone1_mask = tones.tone1_mask
            tone2_mask = tones.tone2_mask
            analysis_filter = tones.analysis_filter

            (
                thd_db,
                thd_pct,
                imd_db,
                imd_pct,
                imd_diff_freq,
                imd_diff_db,
                imd_diff_pct,
                sinad_db,
                sinad_pct,
                snr_db,
                enob_bits,
            ) = compute_metrics(mag, tones, freqs, fmask, cfg)

            if stable_count == write_after and args.csv:
                mode_name = "IMD" if imd_mode else "THD"
                with open(args.csv, "a", encoding="utf-8") as f:
                    f.write(
                        f"{mode_name},{tone1_freq},{tone2_freq},{thd_db},{thd_pct},{imd_db},{imd_pct},"
                        f"{imd_diff_freq},{imd_diff_db},{imd_diff_pct},"
                        f"{sinad_db},{sinad_pct},{snr_db},{enob_bits},{vrms},{prms}\n"
                    )

            plot_freq(ax_freq, freqs, mag, wc, i_lo, i_hi,
                    freq_range, db_range, formatter_hz, formatter_db)

            if not imd_mode and carrier_idx > 0:
                for i in range(1, 1 + cfg.thd_harmonics):
                    idx = carrier_idx * i
                    if idx < len(mag):
                        y = mag[idx]
                        if y > 1e-10:
                            y_db = 20.0 * math.log10(y)
                            if y_db > db_range[0]:
                                ax_freq.text(
                                    freqs[idx],
                                    y_db,
                                    str(i),
                                    horizontalalignment="center",
                                    verticalalignment="bottom",
                                    color="c",
                                    fontstyle="italic",
                                )
            else:
                for tone_freq, label in [(tone1_freq, "F1"), (tone2_freq, "F2")]:
                    idx = nearest_index(freqs, tone_freq)
                    y = mag[idx]
                    if y > 1e-10:
                        y_db = 20.0 * math.log10(y)
                        if y_db > db_range[0]:
                            ax_freq.text(
                                freqs[idx],
                                y_db,
                                label,
                                horizontalalignment="center",
                                verticalalignment="bottom",
                                color="c",
                                fontstyle="italic",
                            )
            
            render_status(skip2, imd_mode,
                  tone1_freq, tone2_freq,
                  thd_db, thd_pct,
                  imd_db, imd_pct,
                  imd_diff_db, imd_diff_pct,
                  sinad_db, sinad_pct,
                  snr_db, enob_bits,
                  vpp, vrms, prms,
                  cfg, best_freqs)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(0.01)

            if stable_count == write_after and pic_base:
                plt.savefig(f"{pic_base}_{tone1_freq:.0f}Hz_{tone2_freq:.0f}Hz{pic_ext}")

        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
        finally:
            audio.terminate()
            plt.close("all")


if __name__ == "__main__":
    raise SystemExit(main())
