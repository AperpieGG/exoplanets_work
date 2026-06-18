import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from plot_images import plot_images
import seaborn as sns

plot_images()


sns.set(
    context="paper",
    style="ticks",
    palette="deep",
    font_scale=1.8,
    color_codes=True
)
# sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
# sns.set_context(rc={"lines.markeredgewidth": 1})

# -------------------------------------------------------
# LOAD CSV
# -------------------------------------------------------
path = '/Users/u5500483/Downloads/'
archive = pd.read_csv(path + "PS_2025_11_20.csv")


# -------------------------------------------------------
# CLEAN: keep only rows with period, radius, mass, eccentricity
# -------------------------------------------------------
df = archive.dropna(
    subset=[
        "pl_orbper",
        "pl_radj",
        "pl_bmassj",
        "pl_orbeccen",
        "st_teff",
        "pl_eqt",
        "sy_kmag"
    ]
)


# -------------------------------------------------------
# APPLY GLOBAL FILTER
# -------------------------------------------------------
df_f = df[
    (df["pl_radj"] > 0.5) &
    (df["pl_orbper"] > 10)
]

print(f"Number of planets after filtering: {len(df_f)}")


# -------------------------------------------------------
# EXTRACT FILTERED COLUMNS
# -------------------------------------------------------
P = df_f["pl_orbper"]
Rj = df_f["pl_radj"]
Mj = df_f["pl_bmassj"]
ecc = df_f["pl_orbeccen"]
st_eff = df_f["st_teff"]
pl_eqt = df_f["pl_eqt"]
Jmag = df_f["sy_kmag"]

# -------------------------------------------------------
# TARGET PARAMETERS
# -------------------------------------------------------
aperpiegg_radious = 1.088
aperpiegg_radious_sigma = 0.012

aperpiegg_ecc = 0.386
aperpiegg_ecc_sigma = 0.019

aperpiegg_period = 58.204720
aperpiegg_period_sigma = 0.00004

aperpiegg_mass = 1.467
aperpiegg_mass_sigma = 0.081

aperpiegg_T_star = 6053
aperpiegg_T_star_sigma = 67

aperpiegg_T_eqt = 519
aperpiegg_T_eqt_sigma = 6.1

aperpiegg_K = 10.197
aperpiegg_K_sigma = 0.026


# -------------------------------------------------------
# COLOURMAP SETTINGS
# -------------------------------------------------------
cmap = plt.get_cmap("cividis")

# For plots colour-coded by planet mass
mass_norm = mcolors.Normalize(vmin=0, vmax=10)
aperpiegg_mass_colour = cmap(mass_norm(aperpiegg_mass))

# For plots colour-coded by eccentricity
ecc_norm = mcolors.Normalize(vmin=0, vmax=0.5)
aperpiegg_ecc_colour = cmap(ecc_norm(aperpiegg_ecc))

# For plots colour-coded by J-band magnitude
jmag_norm = mcolors.Normalize(vmin=8, vmax=12)
aperpiegg_J_colour = cmap(jmag_norm(aperpiegg_K))


# -------------------------------------------------------
# MAKE THE TWO-PANEL FIGURE
# -------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(6, 8))


# -------------------------------------------------------
# 1. PERIOD – RADIUS, colour = MASS
# -------------------------------------------------------
sc1 = axes[0].scatter(
    P,
    Rj,
    c=Mj,
    s=10,
    cmap=cmap,
    edgecolors='black',
    norm=mass_norm,
    alpha=0.7
)

axes[0].errorbar(
    aperpiegg_period,
    aperpiegg_radious,
    xerr=aperpiegg_period_sigma,
    yerr=aperpiegg_radious_sigma,
    fmt='*',
    markersize=15,
    markeredgecolor='black',
    markeredgewidth=1.2,
    color=aperpiegg_mass_colour,
    ecolor='black',
    elinewidth=1.5,
    capsize=5,
    label='Aperpiegg',
    zorder=5
)

axes[0].set_xscale("log")
axes[0].set_xlim(9, 1000)
axes[0].set_xlabel("Orbital Period (days)")
axes[0].set_ylabel("Planet Radius (R$_\\mathrm{Jup}$)")

cbar1 = plt.colorbar(sc1, ax=axes[0], extend='max')
cbar1.set_label("Planet Mass (M$_\\mathrm{Jup}$)")


# -------------------------------------------------------
# 2. ECCENTRICITY – PERIOD, colour = MASS
# -------------------------------------------------------
sc2 = axes[1].scatter(
    P,
    ecc,
    c=Mj,
    s=12,
    cmap=cmap,
    edgecolors='black',
    norm=mass_norm,
    alpha=0.7
)

axes[1].errorbar(
    aperpiegg_period,
    aperpiegg_ecc,
    xerr=aperpiegg_period_sigma,
    yerr=aperpiegg_ecc_sigma,
    fmt='*',
    markersize=15,
    markeredgecolor='black',
    markeredgewidth=1.2,
    color=aperpiegg_mass_colour,
    ecolor='black',
    elinewidth=1.5,
    capsize=5,
    label='Aperpiegg',
    zorder=5
)

axes[1].set_xscale("log")
axes[1].set_xlim(9, 2000)
axes[1].set_xlabel("Orbital Period (days)")
axes[1].set_ylabel("Orbital Eccentricity")

cbar2 = plt.colorbar(sc2, ax=axes[1], extend='max')
cbar2.set_label("Planet Mass (M$_\\mathrm{Jup}$)")


plt.tight_layout()

path_to_save = '/Users/u5500483/Downloads/'
fig.savefig(path_to_save + "period_radius_mass.pdf", bbox_inches="tight")
plt.show()


# -------------------------------------------------------
# STELLAR TEMPERATURE VS PLANET EQUILIBRIUM TEMPERATURE
# colour = ECCENTRICITY
# -------------------------------------------------------
fig2, ax = plt.subplots(figsize=(6, 5))

sc3 = ax.scatter(
    st_eff,
    pl_eqt,
    c=ecc,
    s=12,
    cmap=cmap,
    edgecolors='black',
    norm=ecc_norm,
    alpha=0.7
)

ax.errorbar(
    aperpiegg_T_star,
    aperpiegg_T_eqt,
    xerr=aperpiegg_T_star_sigma,
    yerr=aperpiegg_T_eqt_sigma,
    fmt='*',
    markersize=15,
    markeredgecolor='black',
    markeredgewidth=1.2,
    color=aperpiegg_ecc_colour,
    ecolor='black',
    elinewidth=1.5,
    capsize=5,
    label='Aperpiegg',
    zorder=5
)

ax.set_xlabel("Stellar Effective Temperature (K)")
ax.set_ylabel("Planet Equilibrium Temperature (K)")
ax.set_xlim(2600, 7000)
ax.set_ylim(0, 1500)


# -------------------------------------------------------
# CHEMISTRY TRANSITION LINES
# -------------------------------------------------------
transition_T_N2 = 500

ax.axhline(
    y=transition_T_N2,
    color='blue',
    linestyle='--',
    linewidth=2
)

ax.text(
    4550,
    transition_T_N2,
    r'N$_2$ $\rightarrow$ NH$_3$',
    color='blue',
    fontsize=16,
    ha='center',
    va='center',
    bbox=dict(facecolor='white', edgecolor='none', pad=2)
)


transition_T_CO = 850

ax.axhline(
    y=transition_T_CO,
    color='brown',
    linestyle='--',
    linewidth=2
)

ax.text(
    4550,
    transition_T_CO,
    r'CO $\rightarrow$ CH$_4$',
    color='brown',
    fontsize=16,
    ha='center',
    va='center',
    bbox=dict(facecolor='white', edgecolor='none', pad=2)
)


cbar3 = plt.colorbar(sc3, ax=ax, extend='max')
cbar3.set_label("Orbital Eccentricity")

plt.tight_layout()
fig2.savefig(path_to_save + "chemistry.pdf", bbox_inches="tight")
plt.show()


# -------------------------------------------------------
# COUNT PLANETS ORBITING G-TYPE STARS WITH P > 10 DAYS
# -------------------------------------------------------
df_G = df[df["st_spectype"].str.contains("F", case=False, na=False)]

P_G = df_G["pl_orbper"]
count_long_period = np.sum(P_G > 10)

print(f"Number of planets orbiting F-type stars with period > 10 days: {count_long_period}")


# -------------------------------------------------------
# COUNTS BASED ON THE ACTUAL AXIS LIMITS USED IN EACH PLOT
# -------------------------------------------------------

# Panel 1: Period--Radius plot
# xlim: 9--1000 days
period_radius_mask = (
    (P >= 9) &
    (P <= 1000)
)

n_period_radius = np.sum(period_radius_mask)

print(f"Number of planets shown in Period--Radius plot: {n_period_radius}")


# Panel 2: Eccentricity--Period plot
# xlim: 9--2000 days
ecc_period_mask = (
    (P >= 9) &
    (P <= 2000)
)

n_ecc_period = np.sum(ecc_period_mask)

print(f"Number of planets shown in Eccentricity--Period plot: {n_ecc_period}")


# Chemistry plot: Stellar Teff--Planet Teq plot
# xlim: 2600--7000 K
# ylim: 0--1500 K
chemistry_mask = (
    (st_eff >= 2600) &
    (st_eff <= 7000) &
    (pl_eqt >= 0) &
    (pl_eqt <= 1500)
)

n_chemistry = np.sum(chemistry_mask)

print(f"Number of planets shown in chemistry plot: {n_chemistry}")

# -------------------------------------------------------
# STELLAR TEMPERATURE VS PLANET EQUILIBRIUM TEMPERATURE
# colour = J-band magnitude
# -------------------------------------------------------
fig3, ax = plt.subplots(figsize=(6, 5))

sc4 = ax.scatter(
    st_eff,
    pl_eqt,
    c=Jmag,
    s=12,
    cmap=cmap,
    edgecolors='black',
    norm=jmag_norm,
    alpha=0.7
)

ax.errorbar(
    aperpiegg_T_star,
    aperpiegg_T_eqt,
    xerr=aperpiegg_T_star_sigma,
    yerr=aperpiegg_T_eqt_sigma,
    fmt='*',
    markersize=15,
    markeredgecolor='black',
    markeredgewidth=1.2,
    color=aperpiegg_J_colour,
    ecolor='black',
    elinewidth=1.5,
    capsize=5,
    label='Aperpiegg',
    zorder=5
)

ax.set_xlabel("Stellar Effective Temperature (K)")
ax.set_ylabel("Planet Equilibrium Temperature (K)")
ax.set_xlim(2600, 7000)
ax.set_ylim(0, 1500)

# -------------------------------------------------------
# CHEMISTRY TRANSITION LINES
# -------------------------------------------------------
transition_T_N2 = 500

ax.axhline(
    y=transition_T_N2,
    color='blue',
    linestyle='--',
    linewidth=2
)

ax.text(
    4550,
    transition_T_N2,
    r'N$_2$ $\rightarrow$ NH$_3$',
    color='blue',
    fontsize=16,
    ha='center',
    va='center',
    bbox=dict(facecolor='white', edgecolor='none', pad=2)
)

transition_T_CO = 850

ax.axhline(
    y=transition_T_CO,
    color='brown',
    linestyle='--',
    linewidth=2
)

ax.text(
    4550,
    transition_T_CO,
    r'CO $\rightarrow$ CH$_4$',
    color='brown',
    fontsize=16,
    ha='center',
    va='center',
    bbox=dict(facecolor='white', edgecolor='none', pad=2)
)

cbar4 = plt.colorbar(sc4, ax=ax, extend='both')
cbar4.set_label("K-band Magnitude")

plt.tight_layout()
fig3.savefig(path_to_save + "chemistry_Kmag.pdf", bbox_inches="tight")
plt.show()