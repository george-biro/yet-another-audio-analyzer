# --- audio_backend.py (or inline in your file) ---

import numpy as np
from dataclasses import dataclass
import sounddevice as sd
import platform
import sys

@dataclass
class AudioConfig:
    sample_rate: int
    chunk: int
    skip: int
    device_index: int
    channel_select: int
    channel_count: int
    adc_range: float
    load_ohm: float
    duration_s: int
    window_name: str
    sim_freq: float
    sim_freq2: float
    sim_noise: float
    sim_amp2: float
    sim_hmncs: float
    thd_harmonics: int
    flt_threshold: float
    center_freq_threshold: float
    two_tone_rel_db: float
    peak_min_separation_hz: float


class RingBuffer:
    def __init__(self, size, channels):
        self.buf = np.zeros((size, channels), dtype=np.float32)
        self.size = size
        self.channels = channels
        self.write = 0
        self.filled = 0

    def push(self, data: np.ndarray):
        n = data.shape[0]

        if n >= self.size:
            self.buf[:] = data[-self.size:]
            self.write = 0
            self.filled = self.size
            return

        w = self.write
        end = w + n

        if end <= self.size:
            self.buf[w:end] = data
        else:
            k = self.size - w
            self.buf[w:] = data[:k]
            self.buf[:end % self.size] = data[k:]

        self.write = end % self.size
        self.filled = min(self.size, self.filled + n)

    def get_latest_view(self, n):
        """Return two views (v1, v2) — zero-copy"""
        if self.filled < n:
            return None, None

        start = (self.write - n) % self.size

        if start + n <= self.size:
            return self.buf[start:start+n], None
        else:
            k = self.size - start
            return self.buf[start:], self.buf[:n-k]

def list_sound_devices() -> None:
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"Input Device id {i} - {dev['name']}")

def open_stream(cfg: AudioConfig, ring: RingBuffer):
    if cfg.device_index < 0:
        return None

    device_info = sd.query_devices(cfg.device_index)

    if device_info["max_input_channels"] < 1:
        raise RuntimeError(f"Device '{device_info['name']}' has no input channels.")

    if cfg.channel_count > device_info["max_input_channels"]:
        print(f"Adjusting channels to {device_info['max_input_channels']}")
        cfg.channel_count = device_info["max_input_channels"]

    if cfg.channel_select >= cfg.channel_count:
        raise ValueError("Invalid channel selection")

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        ring.push(indata)   # ZERO COPY write

    stream = sd.InputStream(
        samplerate=cfg.sample_rate,
        device=cfg.device_index,
        channels=cfg.channel_count,
#        dtype="float32",
        callback=callback,
        blocksize=0,
        latency="low",
    )

    stream.start()
    return stream
