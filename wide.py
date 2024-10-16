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

data = np.genfromtxt('t.csv', delimiter=',')

carrier = data[:0]
THD = data[:1]
THD_DB = data[:2]
THD_N = data[:3]
THD_N_DB = data[:4]
SNR = data[:5]
ENOB = data[:6]
Vrms = data[:7]
Prms = data[:8]

