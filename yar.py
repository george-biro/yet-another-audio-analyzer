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
import argparse

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

def carrier(w,w2,f,h):
    ci = np.argmax(w)
    cf = np.array([f[ci]])
    cw = np.array([w[ci]])
    cw2 = np.array([w2[ci]])
    if (ci > 0):
        i = 2 * ci
        n = len(w)
        while (i < n) and (h > 0):
            cf = np.append(cf, f[i])
            cw = np.append(cw, w[i])
            cw2 = np.append(cw2, w2[i])
            i = i + ci
            h = h - 1

    return cw,cw2,cf

def rms(meas):
    return (np.sum(np.square(meas)) / len(meas))**0.5

def thd(cw2):
    if (len(cw2) < 2) or (cw2[0] < 1e-6):
        return float('nan'), float('nan')

    rh = ((np.sum(cw2) - cw2[0]) / (len(cw2) - 1))
    if (rh < 1e-100):
        return float('nan'), float('nan')
    k = (rh / cw2[0])**0.5
    return dbRel(k), (100.*k)

# same a sinad
def thdn(w2,cw2):
    if (len(cw2) < 2) or (len(w2) < 2) or (cw2[0] < 1e-6):
        return float('nan'), float('nan')
    
    # all harmonics except DC
    sn = (np.sum(w2) - w2[0] - cw2[0]) / (len(w2) - 2)
    if (sn < 1e-100):
        return float('nan'), float('nan')

    k = (sn / cw2[0])**0.5
    return dbRel(k), (100.*k)

def snr(w2, cw2):
    if (len(cw2) < 2) or (len(w2) < 2):
        return float('nan')

    sig = np.sum(cw2)
    ns = (np.sum(w2) - w2[0] - sig) / (len(w2) - 1 - len(cw2));
    if (ns < 0):
        return float('nan')

    k = (sig / ns)**0.5
    return dbPow(k)

def enob(sinad):
    return ((sinad - 1.76) / 6.02)

def ifind(arr, val):
    darr = np.absolute(arr - abs(val))
    return darr.argmin()

def isum(w2, lst):
    s = 0.
    for x in lst:
        if (x < len(w2)):
            s = s + w2[x]
    return s

def imd(w2, a, b):
    sig = isum(w2, [ a, b])
    hmc = isum(w2, [ abs(a-b), 2*a, a+b, 2*b, abs(2*a-b), abs(2*b-a), 2*a+b, a+2*b, 3*a, 3*b ])
    k = hmc / sig
    return dbPow(k), (k*100)

def checkImd(iifreq):
    return ((iifreq[0] != iifreq[1]) and (iifreq[0] != 0) and (iifreq[1] != 0))

class CustomHelpFormatter(argparse.HelpFormatter):
    def _get_help_string(self, action):
        help = action.help
        if action.default is not argparse.SUPPRESS:
            help += f' (default: {action.default})'
        return help

# Begin of the program
parser = argparse.ArgumentParser(
                    prog='Yet another Audio analisator',
                    usage='%(prog)s [options]',
                    formatter_class=CustomHelpFormatter)
parser.add_argument("--rload", type=float, default=8, help="Load resistor in ohm")
parser.add_argument("--vrange", type=float, default=100, help="ADC voltage range in volt")
parser.add_argument("--thd", type=int, default=3, help="Number of harmonics for THD calculation")
parser.add_argument("--dev", type=int, default=4, help="Id of sound device")
parser.add_argument("--srate", type=int, default=192000, help="Sample rate")
parser.add_argument("--chunk", type=int, default=65536, help="Chunk size")
parser.add_argument("--ch", type=int, default=0, help="Selected channel")
parser.add_argument("--res", type=int, default=32, help="ADC resolution")
parser.add_argument("--duration", type=int, default=10, help="time to exit")
parser.add_argument("--mono", action='store_true')
parser.add_argument("--list", action='store_true')
args = parser.parse_args()

Rload = args.rload
Vrange = args.vrange    # voltage range of ADC
thdNum = args.thd       # number of harmonics to be checked
ifreq = [ 7000, 60 ]    # intermodulation test frequencies
#ifreq = [ 0, 0 ]

if (args.res == 16):
    iform = pyaudio.paInt16 # 16-bit resolution
    dtype = np.int16
    adc_res = 2**15
elif (args.res == 32):
    iform = pyaudio.paInt32 # 32-bit resolution
    dtype = np.int32
    adc_res = 2**31
else:
    print("Invalid ADC resolution!")
    quit()


#iform = pyaudio.paInt24 # 16-bit resolution
#dtype = np.int24
chsel = args.ch       # 1 channel
chunk = args.chunk      # FFT window size 
samp_rate = args.srate  # sampling rate
duration = round(args.duration * samp_rate / chunk)
dev_index = args.dev    # device index found (see printout)
if args.mono:
    print("MONO mode!")
    chnum = 1
else:
    chnum = 2


# list_cards()
audio = pyaudio.PyAudio()
if args.list:
    list_sound_devices(audio)
    quit()

skip=1024

# create pyaudio stream
stream = audio.open(format = iform,rate = samp_rate,channels = chnum, input_device_index = dev_index,input = True, frames_per_buffer=chunk+skip)

fig, (skip0, ax1, skip1, ax2, skip2) = plt.subplots(5,1,figsize=(16,9), gridspec_kw={'height_ratios': [.1, 2, .01, 6, .2]})
fig.suptitle('Yet another Audio analyseR')
fig.tight_layout()
fig.subplots_adjust(left=.05, bottom=None, right=None, top=None, wspace=None, hspace=None)
skip0.axis("off")
skip1.axis("off")
skip2.axis("off")

N = chunk // 2
flist = abs(np.fft.fftfreq(chunk) * samp_rate)[:N]
fmax = flist[ifind(flist, 50e3)]
iifreq = [ ifind(flist, ifreq[0]),  ifind(flist, ifreq[1]) ]

tmax = chunk / samp_rate * 1000.0
ts = np.linspace(0, tmax, chunk) 

print("carrier,thd,thd db,thdn, thdn db,snr,Vrms,Prms")

try:
    for xx in range(0, duration):

        # record data chunk 
        stream.start_stream()
        data = stream.read(chunk + skip, exception_on_overflow=False)
        stream.stop_stream()
        meas = np.frombuffer(data, dtype=dtype)[skip*chnum:]
        if chnum > 1:
            meas = meas[chsel::2]
        meas = meas * (Vrange / adc_res)

        if len(meas) < 16:
            quit()
        
        ax1.cla()
        ax1.set_title('Time Domain', loc='left')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Amplitude (V)')
        ax1.set_xlim([0, min(tmax, 50)])
        ax1.set_ylim([-5, 5])
        ax1.plot(ts, meas, 'g')
        ax1.grid()

        # compute furie and the related freq values
        w = (np.abs(np.fft.fft(meas))[:N]) * (0.5 / N)
        if (len(w) != len(flist)):
            quit()

        # time domain calculations
        Vpp = np.max(meas) - np.min(meas)
        Ppeak = np.max(np.square(meas)) / Rload
        Vrms = rms(meas)
        Prms = Vrms**2 / Rload

        # freq domain calculations
        w2 = np.square(w)
        cw, cw2, cf = carrier(w, w2, flist, thdNum)
        print("cw ", cw)
        THD, THDP = thd(cw2)
        SINAD, SINADP = thdn(w2,cw2)
        SNR = snr(w2, cw2)
        imdMode = checkImd(iifreq)
        if imdMode:
            IMD, IMDP = imd(w2, iifreq[0], iifreq[1])
        
        if (len(cw) > 2):
            print("%g,%g,%g,%g,%g,%g,%g,%g" % (cf[0], THD, THDP, SINAD, SINADP, SNR, Vrms,Prms))
#displaying
        # manage axles
        ax2.cla()
        ax2.set_title('Frequency Domain', loc='left')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude (dB)')
        ax2.set_xlim([0, fmax])
        ax2.set_ylim([-160, 0])
        ax2.plot(flist, 20*np.log10(w), 'b-')
        if (len(cf) != len(cw)):
            ax2.scatter(cf, 20*np.log10(cw), 'r')
        ax2.grid()
        if (len(cf) > 0):
            t0 = plt.text(0.5, .1, "Base: %5.1fHz" % cf[0], transform=fig.dpi_scale_trans, fontfamily='monospace')

        t1 = plt.text(2.5, .3, "   Vpp: %5.1fV" % Vpp, transform=fig.dpi_scale_trans,  fontfamily='monospace')
        t2 = plt.text(2.5, .1, " Ppeak: %5.1fW" % Ppeak, transform=fig.dpi_scale_trans,  fontfamily='monospace')

        t3 = plt.text(4.5, .3, "  Eff: %5.1fV" % Vrms, transform=fig.dpi_scale_trans, fontfamily='monospace')
        t4 = plt.text(4.5, .1, "E Pwr: %5.1fW" % Prms, transform=fig.dpi_scale_trans, fontfamily='monospace')

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
