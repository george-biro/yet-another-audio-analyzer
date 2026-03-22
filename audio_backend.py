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
import sys
import threading

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
        self.buffers = [
            np.zeros((chunk, channels), dtype=np.float32) for _ in range(depth)
        ]
        self.pos = 0
        self.head = 0
        self.tail = 0
        self.count = 0
        self.overflow = 0
        self.lock = threading.Lock()

    def reset(self):
        self.pos = 0
        self.buffers[self.head].fill(0.0) 

    def drop(self):
        self.overflow += 1
        self.reset()

    def push(self, data: np.ndarray):
        n = len(data)
        m = min(self.chunk - self.pos, n)
        self.buffers[self.head][self.pos:self.pos + m] = data[:m]
        self.pos += m

        if self.pos >= self.chunk:
            # as drop and push called from the same 
            # thread, here the locking is enough!
            with self.lock:
                hnxt = (self.head + 1) % self.depth
                if hnxt != self.tail:
                    self.head = hnxt
                    self.count += 1

                self.reset()
                

    def pop(self):
        with self.lock:
            if self.head == self.tail:
                return None
            rv = self.tail
            self.tail = (self.tail + 1) % self.depth
            self.count -= 1
            return self.buffers[rv].copy()

    def stat(self):
        with self.lock:
            rv = self.overflow
            self.overflow = 0
            return rv, self.count


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

        latency = "low"  # keep low, but buffer is larger

    else:
        if cfg.sample_rate >= 192000:
            blocksize = 8192
        elif cfg.sample_rate >= 96000:
            blocksize = 4096
        else:
            blocksize = 2048

        #latency = 0.05
        latency = "high"

    print(f"[INFO] blocksize={blocksize} latency={latency}")

    last_adc_time = None
    
    def callback(indata, frames, time_info, status):
        nonlocal last_adc_time

        ti = time_info[0]
        t = ti.inputBufferAdcTime

        drop = False
        
        if status:
            drop = True
        elif last_adc_time is not None:
            dt = t - last_adc_time
            expected = frames / cfg.sample_rate
            err = dt - expected
            if abs(err) > 1e-3:
                drop = True

        last_adc_time = t

        if drop:
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
        clip_off=True,
        dither_off=True,
    )

    stream.start()
    cfg.sample_rate = stream.samplerate
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
