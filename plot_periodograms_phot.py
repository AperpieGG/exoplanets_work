import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import batman
from astropy.timeseries import BoxLeastSquares
from plot_images import plot_images

plot_images()

# ==================================
# LOAD DATA
# ==================================
path = "/Users/u5500483/Documents/Github/exoplanets_work/data/"


# ==================================
# LOAD DATA
# ==================================
def load_lc(file):
    df = pd.read_csv(file, comment='#', sep=r'\s+|,', engine='python')
    df.columns = ['time', 'flux', 'flux_err']
    return df


tess = load_lc(path + "TESS.csv")
sg1 = load_lc(path + "ngts_sg1.csv")

data = pd.concat([tess, sg1], ignore_index=True)
data = data.sort_values('time')

time = np.array(data['time'])
flux = np.array(data['flux'])
flux_err = np.array(data['flux_err'])
#
# flux = flux - 1

# ==================================
# BLS SETUP
# ==================================

model = BoxLeastSquares(time, flux, dy=flux_err)

# IMPORTANT: include 58.2 days properly
periods = np.linspace(10, 600, 20000)

# IMPORTANT: realistic durations for long-period planet
durations = np.linspace(0.05, 0.4, 20)

# run BLS
results = model.power(periods, durations)

# best solution
best = np.argmax(results.power)

P = results.period[best]
T0 = results.transit_time[best]
D = results.duration[best]
depth = results.depth[best]

print("\n==============================")
print("BLS RESULT")
print("==============================")
print(f"Period   : {P:.6f} d")
print(f"T0       : {T0:.6f}")
print(f"Duration : {D:.4f} d")
print(f"Depth    : {depth:.6f}")
print("==============================\n")

# ==================================
# PLOT PERIODGRAM
# ==================================
plt.figure()
plt.plot(results.period, results.power, 'k-')
# plt.axvline(58.204721, color='r', linestyle='--', label='True P = 58.2 d')
plt.axvline(P, color='b', linestyle='--', label=f'BLS peak = {P:.2f} d')
plt.xscale('log')
plt.xlabel("Period (days)")
plt.ylabel("BLS power")
plt.legend()
plt.title("BLS Periodogram")
plt.show()



