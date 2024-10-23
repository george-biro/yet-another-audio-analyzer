#! /usr/bin/python3
#
# Yet Another audio analyzeR
#
# Copyright 2024 George Biro
#
# This program is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the 
# Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
# or FITNESS FOR A PARTICULAR PURPOSE. 
# 
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along 
# with this program. If not, see <https://www.gnu.org/licenses/>. 
#
# For noise measurement:
#
# f_in = m / chunk * f_sampling, where m is prime number
#
# 373 / 65536 * 192000 = 1092.8Hz


import math
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
# import sounddevice as sd
import time
import sys
import argparse
import os.path

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

# Determine carrier and harminics
def carrier(wmagnitude, num, fmask):
    cinx = np.argmax(wmagnitude)
    if cinx < 1:
        j = 1
        k = 1
    else:
        j = cinx 
        k = num

    mfundamental = np.zeros(len(wmagnitude))
    mfundamental[cinx] = 1
    mharmonics = np.zeros(len(wmagnitude))
    (mharmonics[cinx::j])[:k] = 1
    mharmonics = mharmonics - mfundamental
    mfundamental = mfundamental * fmask
    mharmonics = mharmonics * fmask
    return cinx, mfundamental, mharmonics

def rms(meas):
    return (np.sum(np.square(meas)) / len(meas))**0.5

def thd_iec(wmagnitude, mharminics):

    vall = np.sum(np.square(wmagnitude))
    vharmonics = np.sum(np.square(wmagnitude * mharmonics))
    if (vall < 1e-100):
        return float('nan'), float('nan')
    
    k = (vharmonics / vall)**0.5
    return dbRel(k), (100.*k)

def thd_ieee(wmagnitude, mfundamental, mharmonics):

    vfundamental = np.sum(np.square(wmagnitude * mfundamental))
    vharmonics = np.sum(np.square(wmagnitude * mharmonics))
    if (vfundamental < 1e-100):
        return float('nan'), float('nan')
    
    k = (vharmonics / vfundamental)**0.5
    return dbRel(k), (100.*k)

# same a sinad
def thdn(wmagnitude, mfundamental, fmask):

    vfundamental = np.sum(np.square(wmagnitude * mfundamental))
    mnoise = fmask - mfundamental
    vnoise = np.sum(np.square(wmagnitude * mnoise))  

    if (vfundamental < 1e-100):
        return float('nan'), float('nan')

    k = ((vnoise / vfundamental)**0.5)
    return dbRel(k), (100.*k)


def snr(wmagnitue, mfundamental, mharmonics, fmask):
    
    vsignal = np.sum(np.square(wmagnitude * mfundamental))
    mnoise = fmask - mfundamental - mharmonics
    vnoise = np.sum(np.square(wmagnitude * mnoise))

    if (vnoise < 1e-100):
        return float('nan'), float('nan')

    k = ((vsignal / vnoise)**0.5) 
    return dbRel(k)


def thdn2(wcomplex, mfundamental, fmask):

    vfundamental = np.sum(np.square(np.fft.irfft(wcomplex * mfundamental)))
    mnsh = fmask - mfundamental
    vnsh = np.sum(np.square(np.fft.irfft(wcomplex * mnsh)))  

    if (vfundamental < 1e-100):
        return float('nan'), float('nan')

    k = ((vnsh / vfundamental)**0.5)
    return dbRel(k), (100.*k)

def snr2(wcomplex, mfundamental, mharmonics, fmask):
    
    msignal = mfundamental + mharmonics
    vsignal = np.sum(np.square(np.fft.irfft(wcomplex * msignal)))
    mnoise = fmask - msignal
    vnoise = np.sum(np.square(np.fft.irfft(wcomplex * mnoise)))

    if (vnoise < 1e-100):
        return float('nan'), float('nan')

    k = ((vsignal / vnoise)**0.5) 
    return dbRel(k)

def enob(sinad):
    return ((sinad - 1.76) / 6.02)

def ifind(arr, val):
    darr = np.absolute(arr - abs(val))
    return darr.argmin()

def checkImd(iifreq):
    return ((iifreq[0] != iifreq[1]) and (iifreq[0] != 0) and (iifreq[1] != 0))

def argAdc(args):
    if (args.adcres == 16):
        iform = pyaudio.paInt16 # 16-bit resolution
        dtype = np.int16
        adcRes = 2**15
    elif (args.adcres == 24):
        iform = pyaudio.paInt32 # 32-bit resolution
        dtype = np.int32
        adcRes = 2**24
    elif (args.adcres == 32):
        iform = pyaudio.paInt32 # 32-bit resolution
        dtype = np.int32
        adcRes = 2**31
    else:
        print("Invalid ADC resolution!")
        quit()
    return iform, dtype, adcRes

def argFFTsize(x):
    return round(x / 2) * 2


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
parser.add_argument("--list", action='store_true')
parser.add_argument("--freq", type=int, default=192000, help="Sample rate")
parser.add_argument("--dev", type=int, default=4, help="Id of sound device")
parser.add_argument("--chsel", type=int, default=1, help="Selected channel")
parser.add_argument("--chnum", type=int, default=2, help="Number of channels")
parser.add_argument("--chunk", type=int, default=32768, help="FFT size")
parser.add_argument("--skip", type=int, default="1024", help="Skip samples")
parser.add_argument("--adcrng", type=float, default=100, help="ADC voltage range")
parser.add_argument("--adcres", type=int, default=32, help="ADC resolution")
parser.add_argument("--vrange", type=float, default=40, help="Display voltage range")
parser.add_argument("--frange", type=float, default="40000", help="displayed frequency range")
parser.add_argument("--trange", type=float, default="10000", help="displayed time range")
parser.add_argument("--wrange", type=float, default="-150", help="FFT range in dB")
parser.add_argument("--rload", type=float, default=8, help="Load resistor in ohm")
parser.add_argument("--thd", type=int, default=3, help="Number of harmonics for THD calculation")
parser.add_argument("--duration", type=int, default=10, help="time to exit")
parser.add_argument("--plot", type=str, default="", help="plot to file")
parser.add_argument("--csv", type=str, default="", help="print to csv")
parser.add_argument("--comment", type=str, default="", help="csv comment")
parser.add_argument("--window", type=str, default="hanning", help="filtering window")
args = parser.parse_args()

Rload = args.rload
Vrange = args.vrange   
Frange = args.frange
Wrange = args.wrange
adcRng = args.adcrng    # voltage range of ADC
thdNum = args.thd       # number of harmonics to be checked
skip = args.skip

iform, dtype, adcRes = argAdc(args)

def clog(wmagnitude):

    wmax = np.max(wmagnitude)
    if (wmax > 1e-6):
        wuni = wmagnitude / wmax
    else:
        wuni = wmagnitude
    wuni[wuni < 1e-6] = 1e-6

    return 20*np.log10(wuni)


def argsMono(x):
    if x:
        print("MONO mode!")
        return 1
    return 2


chSel = args.chsel      
chNum = args.chnum
chunk = argFFTsize(args.chunk)      # FFT window size 
sRate = args.freq  # sampling rate
duration = max(1, round(args.duration * sRate / chunk))
dev_index = args.dev    # device index found (see printout)

audio = pyaudio.PyAudio()
if args.list:
    list_sound_devices(audio)
    quit()

# create pyaudio stream
stream = audio.open(format = iform,rate = sRate,channels = chNum, input_device_index = dev_index,input = True, frames_per_buffer=chunk+skip)

def on_press(event):
    print("on press")
    quit()

def on_close(event):
    print("on close")
    quit()

fig, (skip0, ax1, skip1, ax2, skip2) = plt.subplots(5,1,figsize=(16,9), gridspec_kw={'height_ratios': [.1, 2, .01, 6, .3]})
#fig.canvas.mpl_connect('key_press_event', on_press)
fig.canvas.mpl_connect('close_event', on_close)

fig.suptitle('Yet another Audio analyseR')
fig.tight_layout()
fig.subplots_adjust(left=.05, bottom=None, right=None, top=None, wspace=None, hspace=None)
skip0.axis("off")
skip1.axis("off")
skip2.axis("off")

flist = abs(np.fft.rfftfreq(chunk) * sRate)
# Determine Audio Range
FrangeLow = 20
fmask = np.zeros(len(flist))
ilist = [ ifind(flist, FrangeLow), ifind(flist, Frange) ]
fmask[ilist[0]:ilist[1]] = 1

# Determin Time Range
tmax = chunk / sRate * 1000.0
Trange = min(tmax, args.trange) 
ts = np.linspace(0, tmax, chunk) 

if args.window == "bartlet":
    win = np.bartlet(chunk)
elif args.window == "blackman":
    win = np.blackman(chunk)
elif args.window == "hamming":
    win = np.hamming(chunk)
elif args.window == "hanning":
    win = np.hanning(chunk)
else:
    win = np.ones(chunk)

csvfile = args.csv 

if (csvfile != "") and (not os.path.isfile(csvfile)):
    f = open(csvfile, 'w+')
    f.write("Carrier,THD,THD DB,THD-N,THD-N DB,SNR,ENOB,Vrms,Prms\n")

if args.plot != "":
    picfile, picfile_ext = os.path.splitext(args.plot)
else:
    picfile = ""
    picfile_ext = ""

pCInx = -1
inxCnt = 0
for xx in range(0, duration):

    # record data chunk 
    stream.start_stream()
    data = stream.read(chunk + skip, exception_on_overflow=False)
    stream.stop_stream()
    measFull = np.frombuffer(data, dtype=dtype)[skip*chNum:]
    if chNum > 1:
        measRaw = measFull[chSel::2]
    else:
        measRaw = measFull
    meas = measRaw * win * (adcRng / adcRes) 

    if len(meas) < 16:
        print("len(meas) < 16")
        quit()
        
    ax1.cla()
    ax1.set_title('Time Domain', loc='left')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude (V)')
    ax1.set_xlim([0, Trange])
    ax1.set_ylim([-Vrange, Vrange])
    ax1.plot(ts, meas, 'g')
    ax1.grid()

    # compute furie and the related freq values
    wcomplex = np.fft.rfft(meas) * fmask
    wmagnitude = np.abs(wcomplex)   
    if (len(wmagnitude) != len(flist)):
        print("len(w)%d != len(flist)%d" % (len(wmagnitude), len(flist)))
        quit()

    # time domain calculations
    Vpp = np.max(meas) - np.min(meas)
    Ppeak = np.max(np.square(meas)) / Rload
    Vrms = rms(meas)
    Prms = Vrms**2 / Rload

     # freq domain calculations
    cinx, mfundamental, mharmonics = carrier(wmagnitude, thdNum, fmask)
    if (abs(np.sum(mfundamental) - 1) > .01):
        quit()
    if (np.sum(mharmonics) >= np.sum(fmask)):
        quit()

    THD, THDP = thd_ieee(wmagnitude, mfundamental, mharmonics)
#    SINAD, SINADP = thdn(wmagnitude, mfundamental, fmask)
#    SNR = snr(wmagnitude, mfundamental, mharmonics, fmask)
    SINAD, SINADP = thdn2(wcomplex, mfundamental, fmask)
    SNR = snr2(wcomplex, mfundamental, mharmonics, fmask)
    ENOB = enob(SNR)

#    imdMode = checkImd(iifreq)
#    if imdMode:
#        IMD, IMDP = imd(w2, iifreq[0], iifreq[1])
        
    if (pCInx == cinx):
        inxCnt = inxCnt + 1
    else:
        inxCnt = 0

    ffreq = flist[cinx]
    pCInx = cinx
    if ((inxCnt == 2) and (csvfile != "")):
        f = open(csvfile, "a")
        f.write("%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % (ffreq, THD, THDP, SINAD, SINADP, SNR, ENOB, Vrms, Prms))
#displaying
    # manage axles
    ax2.cla()
    ax2.set_title('Frequency Domain', loc='left')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Amplitude (dB)')
    ax2.set_xlim([FrangeLow, Frange])
    ax2.set_xscale("log")
    ax2.set_ylim([Wrange, 0])

    ax2.plot(flist[ilist[0]:ilist[1]], clog(wmagnitude[ilist[0]:ilist[1]]), 'b-')
# ax2.scatter(cf, 20*np.log10(wa * (mc + mh), 'r')
    ax2.grid()
    t0 = plt.text(0.5, .3, "Base : %5.1fHz" % ffreq, transform=fig.dpi_scale_trans, fontfamily='monospace')
    t7 = plt.text(0.5, .1, "Wsize: %5d#" % chunk, transform=fig.dpi_scale_trans, fontfamily='monospace') 

    t1 = plt.text(2.5, .3, "   Vpp: %5.1fV" % Vpp, transform=fig.dpi_scale_trans,  fontfamily='monospace')
    t2 = plt.text(2.5, .1, " Ppeak: %5.1fW" % Ppeak, transform=fig.dpi_scale_trans,  fontfamily='monospace')

    t3 = plt.text(4.5, .3, "  Eff: %5.1fV" % Vrms, transform=fig.dpi_scale_trans, fontfamily='monospace')
    t4 = plt.text(4.5, .1, "E Pwr: %5.1fW" % Prms, transform=fig.dpi_scale_trans, fontfamily='monospace')

    t5 = plt.text(6.5, .3,  "Range: %3.1fV" % Vrange, transform=fig.dpi_scale_trans, fontfamily='monospace')
    t6 = plt.text(6.5, .1,  " Load: %3.1fohm" % Rload, transform=fig.dpi_scale_trans, fontfamily='monospace') 

    t8 = plt.text(8.5, .3, "THD(%02d): %5.1fdB (%6.3f%%)" % (thdNum, THD, THDP), transform=fig.dpi_scale_trans, fontfamily='monospace') 
    t9 = plt.text(8.5, .1, "  THD-N: %5.1fdB (%6.3f%%)" % (SINAD, SINADP), transform=fig.dpi_scale_trans, fontfamily='monospace')
    t10 = plt.text(11.5, .1, "    SNR: %5.1fdB  ENOB %3.1f" % (SNR, ENOB), transform=fig.dpi_scale_trans, fontfamily='monospace')

#    if imdMode:
#        t11 = plt.text(9, .5, "IMD: %5.1fdB (%4.2f%%)" % (IMD, IMDP), transform=fig.dpi_scale_trans, fontfamily='monospace')
#        t12 = plt.text(9, .3, " F1: %5.1fHz" % flist[iifreq[0]], transform=fig.dpi_scale_trans, fontfamily='monospace') 
#        t13 = plt.text(9, .1, " F2: %5.1fHz" % flist[iifreq[1]], transform=fig.dpi_scale_trans, fontfamily='monospace')
    
    plt.pause(.01)
    if ((inxCnt == 2) and (picfile != "")):
        plt.savefig("%s_%.0fHz%s" % (picfile, ffreq, picfile_ext))

    #if (len(cf) > 0):
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

#    if imdMode:
#        t11.remove()
#        t12.remove()
#        t13.remove()

