#! /usr/bin/python3
#
# Yet Another audio analyzeR
#
#
# Copyright 2024 George Biro
#
# Permission to use, copy, modify, distribute, and sell this 
# software and its documentation for any purpose is hereby 
# granted without fee, provided that the above copyright notice 
# appear in all copies and that both that copyright notice and 
# this permission notice appear in supporting documentation.
#
# The above copyright notice and this permission notice shall be 
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE OPEN GROUP BE LIABLE FOR 
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.
#
#
# Except as contained in this notice, the name of George Biro
# shall not be used in advertising or otherwise to promote the 
# sale, use or other dealings in this Software without prior 
# written authorization from George Biro.
#
#

import math
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
# import sounddevice as sd
import time
import sys

Rload = 8               # load resistor
Vrange = 1              # voltage range of ADC
thdNum = 99             # number of harmonics to be checked
ifreq = [ 7000, 60 ]    # intermodulation test frequencies
#ifreq = [ 0, 0 ]

iform = pyaudio.paInt16 # 16-bit resolution
dtype = np.int16
adc_res = 32768
#iform = pyaudio.paInt24 # 16-bit resolution
#dtype = np.int24
chans = 1               # 1 channel
samp_rate = 44200       # sampling rate
chunk = 32768           # FFT window size 
dev_index = 0          # device index found (see printout)

def list_sound_devices(audio):
    host = 0
    info = audio.get_host_api_info_by_index(host)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(host, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(host, i).get('name'))


def dbRel(k):
    if (k <= 0):
        return float('nan')
    return 20.*math.log10(k)

def dbPow(k):
    if (k <= 0):
        return float('nan')
    return 10.*math.log10(k)

def carrier(w,f,h):
    ci = np.argmax(w)
    cf = np.array([f[ci]])
    cw = np.array([w[ci]])
    if (ci > 0):
        i = ci * 2
        N = len(w)
        while (i < N) and (h > 0):
            cf = np.append(cf, f[i])
            cw = np.append(cw, w[i])
            i = i + ci
            h = h - 1

    return cw,cf

def thd(cw):
    if (len(cw) < 1):
        return float('nan'), float('nan')

    # harmonics
    h = (np.sum(np.square(cw)) - cw[0]**2)**0.5
    k = h / cw[0]
    return dbRel(k), (100.*k)

# same a sinad
def thdn(X,cw):
    if (len(cw) < 1) or (len(X) < 1):
        return float('nan'), float('nan')
    # all harmonics except DC
    sn = (np.sum(np.square(X)) - X[0]**2.0 - cw[0]**2.0)**0.5
    k = sn / cw[0]
    return dbRel(k), (100.*k)

def snr(w, cw):
    if (len(cw) < 1) or (len(w) < 1):
        return float('nan'), float('nan')

    sig = np.sum(np.square(cw))
    ns = np.sum(np.square(w)) - w[0]**2.0 - sig;
    k = sig / ns
    return dbPow(k)

def enob(sinad):
    return ((sinad - 1.76) / 6.02)

def eff(meas):
    return (np.sum(np.square(meas)) / len(meas))**0.5

def ifind(arr, val):
    darr = np.absolute(arr - abs(val))
    return darr.argmin()

def isum(w, lst):
    s = 0.
    for x in lst:
        if (x < len(w)):
            s = s + w[x]**2
    return s

def imd(w, a, b):
    sig = isum(w, [ a, b])
    hmc = isum(w, [ abs(a-b), 2*a, a+b, 2*b, abs(2*a-b), abs(2*b-a), 2*a+b, a+2*b, 3*a, 3*b ])
    k = hmc / sig
    return dbPow(k), (k*100)

def checkImd(iifreq):
    return ((iifreq[0] != iifreq[1]) and (iifreq[0] != 0) and (iifreq[1] != 0))


# list_cards()
audio = pyaudio.PyAudio()
list_sound_devices(audio)

# create pyaudio stream
stream = audio.open(format = iform,rate = samp_rate,channels = chans, input_device_index = dev_index,input = True, frames_per_buffer=chunk)

fig, (skip0, ax1, skip1, ax2, skip2) = plt.subplots(5,1,figsize=(16,9), gridspec_kw={'height_ratios': [.1, 2, .01, 6, .2]})
fig.suptitle('Yet another Audio analyseR')
fig.tight_layout()
fig.subplots_adjust(left=.05, bottom=None, right=None, top=None, wspace=None, hspace=None)
skip0.axis("off")
skip1.axis("off")
skip2.axis("off")
wmax = 0

flist = abs(np.fft.fftfreq(chunk) * samp_rate)
iifreq = [ ifind(flist, ifreq[0]),  ifind(flist, ifreq[1]) ]

try:
    while True:

        # record data chunk 
        stream.start_stream()
        data = stream.read(chunk)
        stream.stop_stream()
        meas = np.frombuffer(data, dtype=dtype) / adc_res * Vrange
        tmax = chunk / samp_rate * 1000.0
        ts = np.linspace(0, tmax, len(meas)) 

        ax1.cla()
        ax1.set_title('Time Domain', loc='left')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Amplitude (V)')
        ax1.set_xlim([0, tmax])
        ax1.set_ylim([-1, 1])
        ax1.plot(ts, meas)
        ax1.grid()

        # compute furie and the related freq values
        w = np.abs(np.fft.fft(meas))
        N = len(w)
        w = w * (1 / N)
     
        N = N//2
    
        # drop second half
        w = w[:N]
        flist = flist[:N]

        # time domain calculations
        peakV = np.max(np.abs(meas))
        peakW = peakV**2 / Rload
        rmsV = eff(meas)
        rmsW = rmsV**2 / Rload

        # freq domain calculations
        cw,cf = carrier(w, flist, thdNum)
 
        THD, THDP = thd(cw)
        SINAD, SINADP = thdn(w,cw)
        SNR = snr(w, cw)
        imdMode = checkImd(iifreq)
        if imdMode:
            IMD, IMDP = imd(w, iifreq[0], iifreq[1])

#displaying
        # manage axles
        ax2.cla()
        ax2.set_title('Frequency Domain', loc='left')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude (V)')
        ax2.set_xlim([0, flist[N-1]])
        wmax = math.floor(max(wmax, np.max(w)) * 100 + 1) / 100
        ax2.set_ylim([0, wmax])
        ax2.plot(flist, w)
        ax2.scatter(cf, cw)
        ax2.grid()
        if (len(cf) > 0):
            t0 = plt.text(0.5, .1, "Base: %5.1fHz" % cf[0], transform=fig.dpi_scale_trans, fontfamily='monospace')

        t1 = plt.text(2.5, .3, "  Peak: %5.1fV" % peakV, transform=fig.dpi_scale_trans,  fontfamily='monospace')
        t2 = plt.text(2.5, .1, "Pk Pwr: %5.1fW" % peakW, transform=fig.dpi_scale_trans,  fontfamily='monospace')

        t3 = plt.text(4.5, .3, "  Eff: %5.1fV" % rmsV, transform=fig.dpi_scale_trans, fontfamily='monospace')
        t4 = plt.text(4.5, .1, "E Pwr: %5.1fW" % rmsW, transform=fig.dpi_scale_trans, fontfamily='monospace')

        t5 = plt.text(6.5, .5,  "Range: %3.1fV" % Vrange, transform=fig.dpi_scale_trans, fontfamily='monospace')
        t6 = plt.text(6.5, .3,  " Load: %3.1fohm" % Rload, transform=fig.dpi_scale_trans, fontfamily='monospace') 
        t7 = plt.text(6.5, .1,  "Wsize: %5d#" % chunk, transform=fig.dpi_scale_trans, fontfamily='monospace') 
    

        t8 = plt.text(11, .5, "THD(%02d): %5.1fdB (%6.3f%%)" % (thdNum, THD, THDP), transform=fig.dpi_scale_trans, fontfamily='monospace') 
        t9 = plt.text(11, .3, "  THD-N: %5.1fdB" % (SINAD), transform=fig.dpi_scale_trans, fontfamily='monospace')
        t10 = plt.text(11, .1, "    SNR: %5.1fdB" % (SNR), transform=fig.dpi_scale_trans, fontfamily='monospace')

        if imdMode:
            t11 = plt.text(9, .5, "IMD: %5.1fdB (%4.2f%%)" % (IMD, IMDP), transform=fig.dpi_scale_trans, fontfamily='monospace')
            t12 = plt.text(9, .3, " F1: %5.1fHz" % flist[iifreq[0]], transform=fig.dpi_scale_trans, fontfamily='monospace') 
            t13 = plt.text(9, .1, " F2: %5.1fHz" % flist[iifreq[1]], transform=fig.dpi_scale_trans, fontfamily='monospace')

        
        plt.pause(.01)

        wmax = wmax * 0.9
        if (len(cf) > 0):
            t0.remove()
        t1.remove()
        t2.remove()
        t3.remove()
        t4.remove()
        t5.remove()
        t6.remove()
        t7.remove()
        t8.remove()
        t9.remove()
        t10.remove()

        if imdMode:
            t11.remove()
            t12.remove()
            t13.remove()


except KeyboardInterrupt:
    pass
