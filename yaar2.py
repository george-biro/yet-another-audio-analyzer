#! /usr/bin/env python3
#
# Yet Another Audio analyzeR
#
# Copyright 2024 George Biro
#
# GPLv3-or-later
#

from __future__ import annotations

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
    sim_noise: float
    thd_harmonics: int
    flt_threshold: float
    fnd_threshold: float
    frq_threshold: float
    center_freq_threshold: float


@dataclass
class AdcFormat:
    pa_format: int
    dtype: np.dtype
    scale: float


def list_sound_devices(audio: pyaudio.PyAudio) -> None:
    host = 0
    info = audio.get_host_api_info_by_index(host)
    num_devices = info.get("deviceCount", 0)
    for i in range(num_devices):
        dev = audio.get_device_info_by_host_api_device_index(host, i)
        if dev.get("maxInputChannels", 0) > 0:
            print(f"Input Device id {i} - {dev.get('name')}")


def riaa_db(freq_hz: float) -> float:
    t1 = 75e-6
    t2 = 318e-6
    t3 = 3180e-6
    w = 2.0 * math.pi * freq_hz
    return (
        10.0 * math.log10(1.0 + 1.0 / ((w * t2) ** 2))
        - 10.0 * math.log10(1.0 + 1.0 / ((w * t1) ** 2))
        - 10.0 * math.log10(1.0 + (w * t3) ** 2)
    )


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


def thd_ieee(wm: np.ndarray, mh: np.ndarray, carrier_idx: int) -> Tuple[float, float]:
    vfund = wm[carrier_idx]
    vharm = np.sum(np.square(wm * mh))
    if vfund < 1e-100:
        return float("nan"), float("nan")
    k = math.sqrt(vharm) / vfund
    return db_rel(k), 100.0 * k


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


def prime_freq_list(freqs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = []
    for i in range(3, len(freqs)):
        if mask[i] > 0.5 and is_prime(i):
            out.append(freqs[i])
    return np.array(out, dtype=float)

def best_freq(prime_list: np.ndarray, freq: float) -> np.ndarray:
    if len(prime_list) == 0:
        return np.array([], dtype=float)

    targets = [60.0, 1000.0, 7000.0, 10000.0, 19000.0, 20000.0]
    indices = [int(np.argmin(np.abs(prime_list - t))) for t in targets]
    return prime_list[np.clip(indices, 0, len(prime_list) - 1)]

def get_adc_format(bits: int) -> AdcFormat:
    if bits == 16:
        return AdcFormat(pyaudio.paInt16, np.int16, 2**15)
    if bits == 24:
        # Stored in 32-bit container for PyAudio compatibility.
        return AdcFormat(pyaudio.paInt32, np.int32, 2**24)
    if bits == 32:
        return AdcFormat(pyaudio.paInt32, np.int32, 2**31)
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


def simulate_signal(freq_hz: float, noise_amp: float, ts: np.ndarray, window: np.ndarray) -> np.ndarray:
    signal = np.sin(2.0 * np.pi * freq_hz * ts + random.random() * np.pi)
    if noise_amp > 1e-6:
        signal = signal + np.random.normal(0.0, noise_amp, len(signal))
    return signal * window


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

    return parser


def init_csv(path: str) -> None:
    if path and not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("Carrier,THD_dB,THD_pct,THDN_dB,THDN_pct,SNR_dB,ENOB,Vrms,Prms\n")


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
        sim_noise=noise_from_db(args.simnoise),
        thd_harmonics=args.thd,
        flt_threshold=from_db(-args.flttsh),
        fnd_threshold=from_db(-args.fndtsh),
        frq_threshold=from_db(-args.frqtsh),
        center_freq_threshold=args.cftsh,
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

        window = get_window(cfg.window_name, cfg.chunk)
        init_csv(args.csv)

        pic_base, pic_ext = os.path.splitext(args.plot) if args.plot else ("", "")

        fig, (skip0, ax_time, skip1, ax_freq, skip2) = plt.subplots(
            5, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [0.1, 2, 0.01, 6, 0.5]}
        )

        closed = {"value": False}

        def on_close(_event):
            closed["value"] = True

        fig.canvas.mpl_connect("close_event", on_close)
        fig.suptitle("Yet Another Audio analyzeR")
        fig.tight_layout()
        fig.subplots_adjust(left=0.06, hspace=None)
        skip0.axis("off")
        skip1.axis("off")
        skip2.axis("off")

        formatter_s = EngFormatter(unit="s")
        formatter_v = EngFormatter(unit="V")
        formatter_hz = EngFormatter(unit="Hz")
        formatter_db = EngFormatter(unit="dB")

        freqs = np.abs(np.fft.rfftfreq(cfg.chunk) * cfg.sample_rate)
        fmask = np.zeros(len(freqs), dtype=float)
        i_lo = nearest_index(freqs, freq_range[0])
        i_hi = nearest_index(freqs, freq_range[1])
        fmask[i_lo:i_hi] = 1.0
        prime_freqs = prime_freq_list(freqs, fmask)

        tmax = cfg.chunk / cfg.sample_rate
        time_range = [
            max((tmax - args.trange * 1e-3) * 0.5, 0.0),
            min((tmax + args.trange * 1e-3) * 0.5, tmax),
        ]
        ts = np.linspace(0.0, tmax, cfg.chunk, endpoint=False)

        riaa_1000 = riaa_db(1000.0)

        wmagsum = np.zeros(len(freqs), dtype=float)
        wmagdiv = 0
        prev_carrier_idx = -1
        stable_count = 0
        write_after = 10 if cfg.avg_enabled else 3

        stream = open_stream(audio, cfg, adc)

        t_start = time.time()
        while (time.time() - t_start < cfg.duration_s) and not closed["value"]:
            if stream is None:
                meas = simulate_signal(cfg.sim_freq, cfg.sim_noise, ts, window)
            else:
                meas = read_measurement(stream, cfg, adc, window)

            if len(meas) < 16:
                raise RuntimeError("Measurement buffer too short.")

            # Time-domain plot
            ax_time.cla()
            ax_time.set_title("Time Domain", loc="left")
            ax_time.set_xlim(time_range)
            ax_time.set_ylim(voltage_range)
            ax_time.xaxis.set_major_formatter(formatter_s)
            ax_time.yaxis.set_major_formatter(formatter_v)
            ax_time.plot(ts, meas, "r")
            ax_time.grid()

            # FFT
            spectrum_complex = np.fft.rfft(meas) * fmask
            mag = normalize_unit(np.abs(spectrum_complex) / len(spectrum_complex))

            if len(mag) != len(freqs):
                raise RuntimeError("Spectrum length mismatch.")

            carrier_idx, fundamental_mask, harmonics_mask = carrier(
                mag, cfg.thd_harmonics, fmask, cfg.fnd_threshold
            )

            if np.sum(harmonics_mask) >= np.sum(fmask):
                raise RuntimeError("Harmonics mask overflowed analysis band.")

            carrier_freq = freqs[carrier_idx]
            fundamental_freq = calc_f_freq(mag, freqs, cfg.frq_threshold)

            if prev_carrier_idx == carrier_idx:
                stable_count += 1
            else:
                stable_count = 0
            prev_carrier_idx = carrier_idx

            chosen_freq = (
                carrier_freq
                if abs(fundamental_freq - carrier_freq) < cfg.center_freq_threshold
                else fundamental_freq
            )

            wc = wclean(ts, window, chosen_freq, cfg.flt_threshold)
            analysis_filter = notch(wc, 1e-20) * fmask

            if cfg.avg_enabled:
                if stable_count < 3:
                    wmagsum[:] = 0.0
                    wmagdiv = 0
                wmagsum += mag
                wmagdiv += 1
                wmagnitude = wmagsum / max(wmagdiv, 1)
            else:
                wmagnitude = mag

            # Time-domain metrics
            vpp = float(np.max(meas) - np.min(meas))
            ppeak = float(np.max(np.square(meas)) / cfg.load_ohm)
            vrms = rms(meas)
            prms = (vrms ** 2) / cfg.load_ohm

            # Frequency-domain metrics
            thd_db, thd_pct = thd_ieee(wmagnitude, harmonics_mask, carrier_idx)
            sinad_db, sinad_pct = thdn(wmagnitude, fundamental_mask, analysis_filter, fmask)
            snr_db = snr(wmagnitude, fundamental_mask, analysis_filter, fmask, harmonics_mask)
            enob_bits = enob(sinad_db)

            if stable_count == write_after and args.csv:
                with open(args.csv, "a", encoding="utf-8") as f:
                    f.write(
                        f"{fundamental_freq},{thd_db},{thd_pct},{sinad_db},{sinad_pct},"
                        f"{snr_db},{enob_bits},{vrms},{prms}\n"
                    )

            # Frequency-domain plot
            ax_freq.cla()
            ax_freq.set_title("Frequency Domain", loc="left")
            ax_freq.set_xscale("log")
            ax_freq.set_xlim(freq_range)
            ax_freq.set_ylim(db_range)
            ax_freq.xaxis.set_major_formatter(formatter_hz)
            ax_freq.yaxis.set_major_formatter(formatter_db)
            ax_freq.plot(freqs[i_lo:i_hi], clean_log(wmagnitude[i_lo:i_hi]), "b-")
            ax_freq.plot(freqs[i_lo:i_hi], clean_log(wc[i_lo:i_hi]), "g.")
            ax_freq.grid()

            if carrier_idx > 0:
                for i in range(1, 1 + cfg.thd_harmonics):
                    idx = carrier_idx * i
                    if idx < len(wmagnitude):
                        y = wmagnitude[idx]
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

            info_text = []
            best = best_freq(prime_freqs, fundamental_freq)
            for i, bf in enumerate(best):
                mark = "*" if abs(chosen_freq - bf) < 1e-6 else " "
                info_text.append(
                    plt.text(
                        0.5 + 1.5 * i,
                        0.5,
                        f"{bf:10.5f}Hz{mark}",
                        transform=fig.dpi_scale_trans,
                        fontfamily="monospace",
                        style="italic",
                    )
                )
                mv = 2.5 * 10 ** ((riaa_db(bf) - riaa_1000) / 20)
                print(f"{bf:10.5f}Hz {mv:.1f}mV")

            info_text.extend(
                [
                    plt.text(0.5, 0.3, f"Base : {fundamental_freq:10.5f}Hz",
                             transform=fig.dpi_scale_trans, fontfamily="monospace", weight="bold"),
                    plt.text(0.5, 0.1, f"#W/sr: {cfg.chunk:5d}#/{cfg.sample_rate * 1e-3:4.1f}kHz",
                             transform=fig.dpi_scale_trans, fontfamily="monospace", weight="bold"),
                    plt.text(2.5, 0.3, f"   Vpp: {vpp:5.1f}V",
                             transform=fig.dpi_scale_trans, fontfamily="monospace", weight="bold"),
                    plt.text(2.5, 0.1, f" Ppeak: {ppeak:5.1f}W",
                             transform=fig.dpi_scale_trans, fontfamily="monospace", weight="bold"),
                    plt.text(4.5, 0.3, f"  Eff: {vrms:5.1f}V",
                             transform=fig.dpi_scale_trans, fontfamily="monospace", weight="bold"),
                    plt.text(4.5, 0.1, f"E Pwr: {prms:5.1f}W",
                             transform=fig.dpi_scale_trans, fontfamily="monospace", weight="bold"),
                    plt.text(6.5, 0.3, f"Range: {voltage_range[1]:3.1f}V",
                             transform=fig.dpi_scale_trans, fontfamily="monospace", weight="bold"),
                    plt.text(6.5, 0.1, f" Load: {cfg.load_ohm:3.1f}ohm",
                             transform=fig.dpi_scale_trans, fontfamily="monospace", weight="bold"),
                    plt.text(8.5, 0.3, f"THD({cfg.thd_harmonics:02d}): {thd_db:5.1f}dB ({thd_pct:6.3f}%)",
                             transform=fig.dpi_scale_trans, fontfamily="monospace", weight="bold"),
                    plt.text(8.5, 0.1, f"  THD-N: {sinad_db:5.1f}dB ({sinad_pct:6.3f}%)",
                             transform=fig.dpi_scale_trans, fontfamily="monospace", weight="bold"),
                    plt.text(11.5, 0.1, f"    SNR: {snr_db:5.1f}dB  ENOB {enob_bits:3.1f}",
                             transform=fig.dpi_scale_trans, fontfamily="monospace", weight="bold"),
                ]
            )

            plt.pause(0.01)

            if stable_count == write_after and pic_base:
                plt.savefig(f"{pic_base}_{fundamental_freq:.0f}Hz{pic_ext}")

            for item in info_text:
                item.remove()

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
