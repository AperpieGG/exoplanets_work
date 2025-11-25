import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_images():
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.minor.left'] = True

    plt.rcParams['font.family'] = 'Times'
    plt.rcParams['font.size'] = 16

    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 16


plot_images()


# -------------------------------------------------------
# LOAD CSV (your file)
# -------------------------------------------------------
# Use: archive = pd.read_csv("yourfile.csv")
path= '/Users/u5500483/Downloads/'
archive = pd.read_csv(path + "PS_2025_11_20.csv")


# -------------------------------------------------------
# CLEAN: Keep only rows with period, radius, mass
# -------------------------------------------------------
df = archive.dropna(subset=["pl_orbper", "pl_radj", "pl_bmassj", "pl_orbeccen"])


# Convert to Jupiter units (if needed)
Rj = df["pl_radj"]       # already in Jupiter radii
Mj = df["pl_bmassj"]     # already in Jupiter masses
P  = df["pl_orbper"]
ecc = df["pl_orbeccen"]
st_eff = df["st_teff"]  # stellar effective temperature
pl_eqt = df["pl_eqt"]    # planet equilibrium temperature
pl_orbsmax = df["pl_orbsmax"]  # semi-major axis in AU
aperpiegg_radious = 1.104
aperpiegg_radious_sigma = 0.057
aperpiegg_ecc = 0.3
aperpiegg_ecc_sigma = 0.041
aperpiegg_period = 58.204723
aperpiegg_period_sigma = 0.000038


# -------------------------------------------------------
# MAKE THE TWO-PANEL FIGURE
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))


# -------------------------------------------------------
# 1️⃣ PERIOD – RADIUS (colour = MASS)
# -------------------------------------------------------
sc1 = axes[0].scatter(P, Rj, c=Mj, s=10, cmap="cividis", alpha=0.7, vmin=0, vmax=10)
# add the aperpiegg here
axes[0].errorbar(aperpiegg_period, aperpiegg_radious,
                 xerr=aperpiegg_period_sigma, yerr=aperpiegg_radious_sigma,
                 fmt='*', markersize=15, markeredgecolor='black', markeredgewidth=1.2, color='red', ecolor='black', elinewidth=1.5, capsize=5, label='Aperpiegg', zorder=5)
axes[0].set_xscale("log")
# axes[0].set_yscale("log")
axes[0].set_xlim(0.15, 1000)
axes[0].set_ylim(-0.1, 2.5)

axes[0].set_xlabel("Orbital Period (days)")
axes[0].set_ylabel("Planet Radius (R$_\\mathrm{Jup}$)")

cbar1 = plt.colorbar(sc1, ax=axes[0], extend='max')
cbar1.set_label("Planet Mass (M$_\\mathrm{Jup}$)")


# -------------------------------------------------------
# 2️⃣ ECCENTRICITY – PERIOD (colour = MASS)
# -------------------------------------------------------
sc2 = axes[1].scatter(P, ecc, c=Mj, s=12, cmap="cividis", alpha=0.7, vmin=0, vmax=10)
# add the aperpiegg here
axes[1].errorbar(aperpiegg_period, aperpiegg_ecc,
                 xerr=aperpiegg_period_sigma, yerr=aperpiegg_ecc_sigma,
                 fmt='*', markersize=15, markeredgecolor='black', markeredgewidth=1.2, color='red', ecolor='black', elinewidth=1.5, capsize=5, label='Aperpiegg', zorder=5)
axes[1].set_xscale("log")
axes[1].set_xlabel("Orbital Period (days)")
axes[1].set_ylabel("Orbital Eccentricity")
axes[1].set_xlim(0.2, 2000)
cbar2 = plt.colorbar(sc2, ax=axes[1], extend='max')
cbar2.set_label("Planet Mass (M$_\\mathrm{Jup}$)")


plt.tight_layout()
plt.show()


# -------------------------------------------------------
# plot temperature of start vs temprature of planet
aperpiegg_T_star = 5985
aperpiegg_T_eqt = 514
plt.figure(figsize=(7, 6))
plt.scatter(st_eff, pl_eqt, c=pl_orbsmax, s=12, cmap="cividis", alpha=0.7, vmin=0, vmax=1)
plt.errorbar(aperpiegg_T_star, aperpiegg_T_eqt,
             xerr=100, yerr=50,
             fmt='*', markersize=15, markeredgecolor='black', markeredgewidth=1.2, color='red', ecolor='black', elinewidth=1.5, capsize=5, label='Aperpiegg', zorder=5)
plt.xlabel("Stellar Effective Temperature (K)")
plt.ylabel("Planet Equilibrium Temperature (K)")
cbar = plt.colorbar(extend='max')
plt.xlim(2600, 7000)
plt.ylim(0, 3000)
cbar.set_label("Semi-Major Axis (AU)")
plt.tight_layout()
plt.show()


df_K = df[df["st_spectype"].str.contains("G", case=False, na=False)]

P_K = df_K["pl_orbper"]
Rj_K = df_K["pl_radj"]
Mj_K = df_K["pl_bmassj"]
ecc_K = df_K["pl_orbeccen"]


plt.figure(figsize=(7, 6))
plt.scatter(P_K, Rj_K, c=ecc_K, s=15, cmap="cividis", alpha=0.7, vmin=0, vmax=0.5)

plt.xscale("log")
plt.xlabel("Orbital Period (days)")
plt.ylabel("Planet Radius (R$_\\mathrm{Jup}$)")
plt.errorbar(aperpiegg_period, aperpiegg_radious,
             xerr=aperpiegg_period_sigma, yerr=aperpiegg_radious_sigma,
             fmt='*', markersize=15, markeredgecolor='black', markeredgewidth=1.2, color='red', ecolor='black', elinewidth=1.5, capsize=5, label='Aperpiegg', zorder=5)
plt.title("Planets Orbiting G-type Stars")

cbar = plt.colorbar()
cbar.set_label("Orbital Eccentricity")

plt.xlim(0.15, 1000)
plt.ylim(-0.1, 2.5)
plt.tight_layout()
plt.show()