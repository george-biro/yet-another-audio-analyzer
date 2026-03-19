#! /usr/bin/env python3
#
# Yet Another Audio analyzeR
#
# Copyright 2026 George Biro
#
# GPLv3-or-later
# 
# File: audio_backend.py

import numpy as np
from dataclasses import dataclass
import sounddevice as sd
import platform
import time
import sys

class MyFreqMeas:
    def __init__(self, num):
        self.last = time.time()
        self.num = num
        self.count = self.num
        self.dt = 0

    def update(self):
        self.count -= 1
        if self.count <= 0:
            now = time.time()
            self.dt = (now - self.last) / self.num
            self.last = now
            self.count = self.num

    def stat(self):
        return self.dt

@dataclass
class AudioConfig:
    sample_rate: int
    chunk: int
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
    def __init__(self, chunk, channels, depth=8):
        self.chunk = chunk
        self.channels = channels
        self.depth = depth

        # FIFO elements are whole chunks
        self.buffers = [
            np.empty((chunk, channels), dtype=np.float32)
            for _ in range(depth)
        ]

        self.pos = 0
        self.head = 0
        self.tail = 0
        self.count = 0
        self.overflow = 0

    def drop(self):
        self.overflow += 1
        self.pos = 0

    def push(self, data: np.ndarray):
        n = len(data)
        m = min(self.chunk - self.pos, n)
        self.buffers[self.head][self.pos:self.pos+m] = data[:m]
        self.pos += m
        if self.pos >= self.chunk:
            self.pos = 0
            self.head = (self.head + 1) % self.depth
            self.count += 1

    def pop(self):
        if self.head == self.tail:
            return None

        rv = self.tail
        self.tail = (self.tail + 1) % self.depth
        self.count -= 1
        if self.count > 3:
            print("WARNING: sys overload!")
        return self.buffers[rv]

    def stat(self):
        return self.overflow

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

    system = platform.system()

    # 🔥 CRITICAL FIX
    if system == "Linux":
        if cfg.sample_rate >= 192000:
            blocksize = 8192
        elif cfg.sample_rate >= 96000:
            blocksize = 4096
        else:
            blocksize = 1024

        latency = "low"   # keep low, but buffer is larger

    else:
        # macOS → keep your fast behavior
        blocksize = 0
        latency = "low"

    print(f"[INFO] blocksize={blocksize} latency={latency}")

    def callback(indata, frames, time_info, status):
        if status:
#            print("AUDIO ERROR:", status, "frames:", frames)
            ring.drop()
        else:
            ring.push(indata)


    stream = sd.InputStream(
        samplerate=cfg.sample_rate,
        device=cfg.device_index,
        channels=cfg.channel_count,
        callback=callback,
        blocksize=blocksize,
        latency=latency,
        dtype="float32",
    )

    stream.start()
    cfg.samplerate = stream.samplerate
    return stream

def read_measurement(cfg, ring):
    
    data = ring.pop()
    if data is None:
        return None
   
#    left = data[:, 0]
#    right = data[:, 1]
#    print(
#        "L rms:", np.sqrt(np.mean(left**2)),
#        "R rms:", np.sqrt(np.mean(right**2))
#    )

    return data[:, cfg.channel_select] * cfg.adc_range
