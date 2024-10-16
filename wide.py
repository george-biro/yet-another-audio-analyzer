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

import numpy as np
import os.path
import argparse
import matplotlib.pyplot as plt

def myplot(x, y, title, xlabel, ylabel, fn):
    
    if (len(x) == 0) or (len(y) == 0):
        print("empty data!")
        print(x)
        print(y)
        quit()
    
    fig, ax = plt.subplots(1,1, figsize=(16,9))
    ax.cla()
    ax.set_title(title, loc='left')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([np.min(x), np.max(x)])
    ax.set_ylim([np.min(y), np.max(y)])
    ax.plot(x, y);
    ax.grid()
    plt.pause(.01)
    plt.savefig(fn)

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
parser.add_argument("--csv", type=str, default="", help="Input CSV file created by yar.py")
parser.add_argument("--plot", type=str, default="", help="Generated plot file")
args = parser.parse_args()

if (args.csv == ""):
    print("Undefined input CSV!")
    quit()

if (args.plot == ""):
    print("Undefined output plot!")
    quit()

dataa = np.genfromtxt(str(args.csv), delimiter=',')
data = dataa[~np.isnan(dataa).any(axis=1)]
carrier = data[:,0]
THD = data[:,1]
THDDB = data[:,2]
THDN = data[:,3]
THDNDB = data[:,4]
SNR = data[:,5]
VRMS = data[:,6]
PRMS = data[:,7]
i = np.argsort(carrier)
carrier = carrier[i]
THD = THD[i]
THDDB = THDDB[i]
THDN = THDN[i]
THDNDB = THDNDB[i]
SNR = SNR[i]
VRMS = VRMS[i]
PRMS = PRMS[i]

pltfile,pltfile_ext = os.path.splitext(args.plot)
myplot(carrier, THD, "THD", "Freq Hz", "%", ("%s-THD.%s" % (pltfile, pltfile_ext)))
myplot(carrier, THDDB, "THD", "Freq Hz", "dB", ("%s-THDDB.%s" % (pltfile, pltfile_ext)))
myplot(carrier, THDN, "THD-N", "Freq Hz", "%", "%s-THDN.%s" % (pltfile, pltfile_ext))
myplot(carrier, THDNDB, "THD-N", "Freq Hz", "dB", "%s-THDNDB.%s" % (pltfile, pltfile_ext))
myplot(carrier, SNR, "snr", "Feq Hz", "dB", "%s-SNR.%s" % (pltfile, pltfile_ext))
myplot(carrier, VRMS, "Vrms", "Freq Hz", "V", "%s-VRMS.%s" % (pltfile, pltfile_ext))
myplot(carrier, PRMS, "Prms", "Freq Hz", "W", "%s-PRMS.%s" % (pltfile, pltfile_ext))


