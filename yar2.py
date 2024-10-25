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
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.
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
import random

# display the list of sound device
def list_sound_devices(audio):
    host = 0
    info = audio.get_host_api_info_by_index(host)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(host, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(host, i).get('name'))

# given number is prime
#
# param n
#   input value
# return
#   true if the number is prime
def isPrime(n):
    for i in range(2, int(n**0.5) + 1):
        if (n % i == 0):
            return True
            
    return False

# find pime numbers around a given number
#
# To find the bigest smallest prime number and and 
# smallest greater number. So for exmlpe the call
#
# findPrime(9) -> 7,11
#
# param n
#   input value
# return
#   returns the two prime numbers
def findPrime(n):
    for i in range(n-1,0,-1):
        if isPrime(i):
            plow = i
            break

    for i in range(n+1, n+100):
        if isPrime(i):
            phigh = i
            break

    return plow, phigh

# Compute relative dB value
def dbRel(k):
    if (k <= 0):
        return float('nan')
    return 20.*math.log10(k)

# Compute power dB value
def dbPow(k):
    if (k <= 0):
        return float('nan')
    return 10.*math.log10(k)

# Notch filter mask
#
# param wc
#   FFT signal magnitude
# param lev
#   requested minimum level
# return
#   An mask array, elements where wc is greater than lev
def notch(wc, lev):
    rv = np.zeros(len(wc))
    rv[wc > lev] = 1
    return rv 

# Determine carrier and harminics
def carrier(w, num, fm, lev):
    cinx = np.argmax(w)
    if cinx < 1:
        j = 1
        k = 1
    else:
        j = cinx 
        k = num

    mfundamental = notch(w, w[cinx] * lev)
    mfundamental[w > (w[cinx]  * lev)] = 1

    mharmonics = np.zeros(len(w))
    (mharmonics[cinx::j])[:k] = 1
    mharmonics[cinx] = 0
    mharmonics = mharmonics * fm
    return cinx, mfundamental, mharmonics

def calcFFreq(w, fl, lev):
    w0 = notch(w, lev) * w
    return np.sum(w0 * fl) / np.sum(w0)

def wclean(ts, win, cfreq, flev):
    mclean = np.sin(2 * np.pi * cfreq * ts / 1000) * win
    wcc = np.fft.rfft(mclean)
    wc = cuni(np.abs(wcc) / len(wcc))
    wc[wc < flev] = 0
    return wc


def rms(meas):
    return (np.sum(np.square(meas)) / len(meas))**0.5

def thd_iec(wmagnitude, mharminics):

    vall = np.sum(np.square(wmagnitude))
    vharmonics = np.sum(np.square(wmagnitude * mharmonics))
    if (vall < 1e-100):
        return float('nan'), float('nan')
    
    k = (vharmonics / vall)**0.5
    return dbRel(k), (100.*k)

def thd_ieee(wmagnitude, mharmonics, cinx):
    vfundamental = wmagnitude[cinx]
    vharmonics = np.sum(np.square(wmagnitude * mharmonics))
    if (vfundamental < 1e-100):
        return float('nan'), float('nan')
    
    k = (vharmonics**.5) / vfundamental
    return dbRel(k), (100.*k)

# same a sinad
def thdn(wm, mfd, mfl, fm):

    vfundamental = np.sum(np.square(wm * mfd))
    mnoise = fm - mfl
    vnoise = np.sum(np.square(wm * mnoise)) 
    
    if (vfundamental < 1e-100):
        return float('nan'), float('nan')

    k = (vnoise / vfundamental)**.5
    return dbRel(k), (100.*k)

def snr(wm, mfd, mflt, fm, mh):
    
    vsignal = np.sum(np.square(wm * mfd))    
    mnoise = fm - mflt - mh
    vnoise = np.sum(np.square(wm * mnoise))
    if (vnoise < 1e-100):
        return float('nan')

    k = (vsignal / vnoise)**.5
    return dbRel(k)

def enob(sinad):
    return ((sinad - 1.76) / 6.02)

def ifind(arr, val):
    darr = np.absolute(arr - abs(val))
    return darr.argmin()

def checkImd(iifreq):
    return ((iifreq[0] != iifreq[1]) and (iifreq[0] != 0) and (iifreq[1] != 0))

def cuni(w):
    wmax = np.max(w)
    if (wmax > 1e-6):
        wuni = w / wmax
    else:
        wuni = w

    return wuni

def clog(wuni2):
    wuni = wuni2
    wuni[wuni < 1e-10] = 1e-10
    return 20*np.log10(wuni)

def simSig(sFreq, sNoise, w):
    r = np.sin(2 * np.pi * sFreq * ts / 1000 + random.random() * np.pi)
    if (sNoise > 1e-6):
            r = r + np.random.normal(0, sNoise, len(r))

    return r * w

#    x = mfund
#    if cinx > 0:
#        x[cinx - 1] = 1
#    
#    cinxr = cinx + 1
#    if cinxr < len(w):
#        x[cinxr] = 1
#
#    y = w * x
#    return np.sum(fl * y) / np.sum(y)

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
    return 2**math.floor(math.log2(x) + .5)

def argsMono(x):
    if x:
        print("MONO mode!")
        return 1
    return 2

def argNoise(x):
    if (x < 0):
        return 0

    return 10**(x / -20)

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
parser.add_argument("--vrange", type=float, default=40, help="Display voltage range in V")
parser.add_argument("--frange", type=float, default="40000", help="displayed frequency range in Hz")
parser.add_argument("--trange", type=float, default="100", help="displayed time range in ms")
parser.add_argument("--wrange", type=float, default="-150", help="FFT range in dB")
parser.add_argument("--rload", type=float, default=8, help="Load resistor in Ohm")
parser.add_argument("--thd", type=int, default=7, help="Number of harmonics for THD calculation")
parser.add_argument("--duration", type=int, default=240, help="time to exit in s")
parser.add_argument("--plot", type=str, default="", help="plot to file")
parser.add_argument("--csv", type=str, default="", help="print to csv")
parser.add_argument("--window", type=str, default="hanning", help="filtering window")
parser.add_argument("--avg", action='store_true', help="Enable avg calculation")
parser.add_argument("--simfreq", type=float, default=0, help="Do sumilation with an exact frequency in Hz")
parser.add_argument("--simnoise", type=float, default=-1, help="Do simulation with exact noise (in dB) amplitude")
parser.add_argument("--flttsh", type=float, default=120, help="Notch filter level in dB to skip around the fundamental signal")
parser.add_argument("--fndtsh", type=float, default=3, help="Filter level to determine the fundamental voltage in dB")
parser.add_argument("--frqtsh", type=float, default=40, help="Filter level to determine the fundamental frequency in dB")
parser.add_argument("--cftsh", type=float, default=.25, help="Center frequency treshold in Hz")
args = parser.parse_args()

doAvg = args.avg
Rload = args.rload
Vrange = args.vrange   
Frange = args.frange
Wrange = args.wrange
adcRng = args.adcrng    # voltage range of ADC
thdNum = args.thd       # number of harmonics to be checked
skip = args.skip
simFreq = args.simfreq
simNoise = argNoise(args.simnoise) 
fltTsh = 10**(args.flttsh / -20)
fndTsh = 10**(args.fndtsh / -20)
frqTsh = 10**(args.frqtsh / -20)
cfTsh = args.cftsh

iform, dtype, adcRes = argAdc(args)

chSel = args.chsel      
chNum = args.chnum
chunk = argFFTsize(args.chunk)      # FFT window size 
sRate = args.freq  # sampling rate
duration = args.duration
dev_index = args.dev    # device index found (see printout)

audio = pyaudio.PyAudio()
if args.list:
    list_sound_devices(audio)
    quit()

# create pyaudio stream
if (dev_index >= 0) and (simFreq < .01):
    stream = audio.open(format = iform,rate = sRate,channels = chNum, input_device_index = dev_index,input = True, frames_per_buffer=chunk+skip)
else:
    stream = None

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
wmagsum = np.zeros(len(flist))
wmagdiv = 0

# Determin Time Range
tmax = chunk / sRate * 1000.0
Trange = [ max((tmax - args.trange) * .5, 0), min((tmax + args.trange) * .5, tmax) ]
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
    f.close()

if args.plot != "":
    picfile, picfile_ext = os.path.splitext(args.plot)
else:
    picfile = ""
    picfile_ext = ""

pCInx = -1
iCnt = 0
tsStart = time.time()

wrCnt = 10 if doAvg else 3

while (time.time() - tsStart < duration):
   
    # record data chunk
    if stream is None:
        meas = simSig(simFreq, simNoise, win)
    else:
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
    ax1.set_xlim(Trange)
    ax1.set_ylim([-Vrange, Vrange])
    ax1.plot(ts, meas, 'r')
    ax1.grid()

    # compute furie and the related freq values
    wcomplex = np.fft.rfft(meas) * fmask
    wmag1 = cuni(np.abs(wcomplex) / len(wcomplex))
    if (len(wmag1) != len(flist)):
        print("len(w)%d != len(flist)%d" % (len(wmag1), len(flist)))
        quit()
    
    cinx, mfundamental, mharmonics = carrier(wmag1, thdNum, fmask, fndTsh)
    if (np.sum(mharmonics) >= np.sum(fmask)):
        quit()

    cfreq = flist[cinx]                                   # center frequency
    ffreq = calcFFreq(wmag1, flist, frqTsh)   # fundamental frequency
    
    if (pCInx == cinx):
        iCnt = iCnt + 1
    else:
        iCnt = 0
    
    pCInx = cinx
    useCFreq = (abs(ffreq - cfreq) < cfTsh)
    wc = wclean(ts, win, cfreq if useCFreq else ffreq, fltTsh)    
    mfilter = notch(wc, 1e-20) * fmask

    if doAvg:
        if (iCnt < 3):
            wmagsum = np.zeros(len(flist))
            wmagdiv = 0

        wmagsum = wmagsum + wmag1
        wmagdiv = wmagdiv + 1
        wmagnitude = wmagsum / wmagdiv
    else:
        wmagnitude = wmag1

    # time domain calculations
    Vpp = np.max(meas) - np.min(meas)
    Ppeak = np.max(np.square(meas)) / Rload
    Vrms = rms(meas)
    Prms = Vrms**2 / Rload

     # freq domain calculations

    THD, THDP = thd_ieee(wmagnitude, mharmonics, cinx)
    SINAD, SINADP = thdn(wmagnitude, mfundamental, mfilter, fmask)
    SNR = snr(wmagnitude, mfundamental, mfilter, fmask, mharmonics)
#    SINAD, SINADP = thdn2(wcomplex, mfundamental, fmask)
#    SNR = snr2(wcomplex, mfundamental, mharmonics, fmask)
    ENOB = enob(SNR)

#    imdMode = checkImd(iifreq)
#    if imdMode:
#        IMD, IMDP = imd(w2, iifreq[0], iifreq[1])

    if ((iCnt == wrCnt) and (csvfile != "")):
        print("write file %s" % csvfile)
        f = open(csvfile, "a")
        f.write("%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % (ffreq, THD, THDP, SINAD, SINADP, SNR, ENOB, Vrms, Prms))
        f.close()
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
    if wc is not None:
        ax2.plot(flist[ilist[0]:ilist[1]], clog(wc[ilist[0]:ilist[1]]), 'g.')

# ax2.scatter(cf, 20*np.log10(wa * (mc + mh), 'r')
    ax2.grid()

    mul = math.floor(sRate / cfreq + .5)
    if isPrime(mul):
        ch = '*' if useCFreq else ' '
        t12 = plt.text(.5, .5, "CFreq: %10.5fHz%c" % 
                (cfreq, ch), transform=fig.dpi_scale_trans, fontfamily='monospace')
    else:
        pl, ph = findPrime(mul)
        fl = cfreq * float(pl) / float(mul)
        fh = cfreq * float(ph) / float(mul)
        t12 = plt.text(.5, .5, "CFreq: %10.5fHz or %10.5fHz" % (fl, fh), transform=fig.dpi_scale_trans, fontfamily='monospace')

    t0 = plt.text(.5, .3, "Base : %10.5fHz" % ffreq, transform=fig.dpi_scale_trans, fontfamily='monospace')
  

    t7 = plt.text(.5, .1, "Wsize: %5d#" % chunk, transform=fig.dpi_scale_trans, fontfamily='monospace') 

    t1 = plt.text(2.5, .3, "   Vpp: %5.1fV" % Vpp, transform=fig.dpi_scale_trans,  fontfamily='monospace')
    t2 = plt.text(2.5, .1, " Ppeak: %5.1fW" % Ppeak, transform=fig.dpi_scale_trans,  fontfamily='monospace')

    t3 = plt.text(4.5, .3, "  Eff: %5.1fV" % Vrms, transform=fig.dpi_scale_trans, fontfamily='monospace')
    t4 = plt.text(4.5, .1, "E Pwr: %5.1fW" % Prms, transform=fig.dpi_scale_trans, fontfamily='monospace')

    t5 = plt.text(6.5, .3,  "Range: %3.1fV" % Vrange, transform=fig.dpi_scale_trans, fontfamily='monospace')
    t6 = plt.text(6.5, .1,  " Load: %3.1fohm" % Rload, transform=fig.dpi_scale_trans, fontfamily='monospace') 

    t8 = plt.text(8.5, .3, "THD(%02d): %5.1fdB (%6.3f%%)" % (thdNum, THD, THDP), transform=fig.dpi_scale_trans, fontfamily='monospace') 
    t9 = plt.text(8.5, .1, "  THD-N: %5.1fdB (%6.3f%%)" % (SINAD, SINADP), transform=fig.dpi_scale_trans, fontfamily='monospace')
    t10 = plt.text(11.5, .1, "    SNR: %5.1fdB  ENOB %3.1f" % (SNR, ENOB), transform=fig.dpi_scale_trans, fontfamily='monospace')
    
    t11 = plt.text(11.5, .3, "   Rate: %5.0fHz" % (sRate), transform=fig.dpi_scale_trans, fontfamily='monospace')

#    if imdMode:
#        t11 = plt.text(9, .5, "IMD: %5.1fdB (%4.2f%%)" % (IMD, IMDP), transform=fig.dpi_scale_trans, fontfamily='monospace')
#        t12 = plt.text(9, .3, " F1: %5.1fHz" % flist[iifreq[0]], transform=fig.dpi_scale_trans, fontfamily='monospace') 
#        t13 = plt.text(9, .1, " F2: %5.1fHz" % flist[iifreq[1]], transform=fig.dpi_scale_trans, fontfamily='monospace')
    
    plt.pause(.01)
    if ((iCnt == wrCnt) and (picfile != "")):
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
    t11.remove()
    t12.remove()
#    if imdMode:
#        t11.remove()
#        t12.remove()
#        t13.remove()

