#!/usr/bin/env python3
#
# Test Generator
#
# Copyright 2026 George Biro
#
# GPLv3-or-later
#
# File: gen.py
#
# This is part of the yaar project
#
import argparse
import math
import numpy as np
import pyaudio
import signal
import sys
import time

TWOPI = 2 * math.pi


def list_devices(audio):
    host = audio.get_host_api_info_by_index(0)
    num = host.get("deviceCount")

    for i in range(num):
        dev = audio.get_device_info_by_host_api_device_index(0, i)

        if dev.get("maxOutputChannels") > 0:
            print(f"Output Device {i}: {dev.get('name')}")


class SineGenerator:
    """
    Professional continuous sine generator.

    Phase accumulator method:
    stable frequency
    no drift
    no discontinuities
    """

    def __init__(self, fs, f1, f2=0.0, ratio=1.0, level=0.5):
        self.fs = fs
        self.f1 = f1
        self.f2 = f2
        self.level = level

        self.phase1 = 0.0
        self.phase2 = 0.0

        self.step1 = TWOPI * f1 / fs
        self.step2 = TWOPI * f2 / fs if f2 > 0 else 0

        self.amp1 = 1.0
        self.amp2 = self.amp1 / ratio if f2 > 0 else 0.0

    def generate(self, n):

        out = np.empty(n, dtype=np.float32)

        p1 = self.phase1
        p2 = self.phase2

        s1 = self.step1
        s2 = self.step2

        a1 = self.amp1
        a2 = self.amp2

        for i in range(n):

            v = a1 * math.sin(p1)

            if self.f2 > 0:
                v += a2 * math.sin(p2)

            out[i] = v * self.level

            p1 += s1
            if p1 > TWOPI:
                p1 -= TWOPI

            if self.f2 > 0:
                p2 += s2
                if p2 > TWOPI:
                    p2 -= TWOPI

        self.phase1 = p1
        self.phase2 = p2

        return out


def main():

    parser = argparse.ArgumentParser("Audio Generator")

    parser.add_argument("--list", action="store_true")

    parser.add_argument("--dev", type=int, default=-1,
                        help="output device")

    parser.add_argument("--rate", type=int, default=192000)

    parser.add_argument("--chunk", type=int, default=1024)

    parser.add_argument("--freq", type=float,
                        help="frequency A")

    parser.add_argument("--freq2", type=float, default=0,
                        help="frequency B")

    parser.add_argument("--ratio", type=float, default=1.0,
                        help="A/B amplitude ratio")

    parser.add_argument("--level", type=float, default=0.5,
                        help="output level (0..1)")

    parser.add_argument(
        "--time",
        type=float,
        default=0,
        help="run time in seconds (0 = infinite)"
    )

    args = parser.parse_args()
    audio = pyaudio.PyAudio()

    if args.list:
        list_devices(audio)
        return

    if args.freq is None:
        parser.error("--freq is required unless --list is used")
        return

    if args.dev < 0:
        parser.error("select output device with --dev")
        return

    stream = audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=args.rate,
        output=True,
        frames_per_buffer=args.chunk,
        output_device_index=args.dev,
    )

    gen = SineGenerator(
        args.rate,
        args.freq,
        args.freq2,
        args.ratio,
        args.level
    )

    running = True

    def stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)

    print("Generator running")
    print(f"Fs: {args.rate} Hz")

    if args.freq2 > 0:
        print(f"Tone A: {args.freq} Hz")
        print(f"Tone B: {args.freq2} Hz")
        print(f"Ratio A/B: {args.ratio}")
    else:
        print(f"Tone: {args.freq} Hz")

    start_time = time.time()

    while running:

        if args.time > 0 and (time.time() - start_time) >= args.time:
            break

        block = gen.generate(args.chunk)
        stream.write(block.tobytes())

    stream.stop_stream()
    stream.close()
    audio.terminate()


if __name__ == "__main__":
    main()
