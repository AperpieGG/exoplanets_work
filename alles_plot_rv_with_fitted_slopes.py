import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import allesfitter

from plot_images import plot_images


RV_SCALE = 1000.0  # km/s -> m/s
DATA_DIR = "data/0205"

RV_INSTRUMENTS = ["HARPS", "CORALIE_1", "CORALIE_2"]

# If the RV slopes are coupled, this is the parameter whose posterior samples
# should be used as the shared uncertainty.
SHARED_RV_SLOPE_KEY = "baseline_slope_rv_HARPS"

DISPLAY_NAMES = {
    "HARPS": "HARPS",
    "CORALIE_1": "CORALIE14",
    "CORALIE_2": "CORALIE24",
}

MARKERS = {
    "HARPS": "o",
    "CORALIE_1": "^",
    "CORALIE_2": "s",
}

COLORS = {
    "HARPS": "blue",
    "CORALIE_1": "orange",
    "CORALIE_2": "green",
}


def get_rv_jitter(inst, posterior_params_median):
    """
    Return RV jitter in native allesfitter units, assumed to be km/s.
    """
    key = "jitter_rv_" + inst
    if key in posterior_params_median:
        return posterior_params_median[key]

    ln_key = "ln_jitter_rv_" + inst
    if ln_key in posterior_params_median:
        return np.exp(posterior_params_median[ln_key])

    return 0.0


def get_posterior_sigma(key, posterior_samples):
    """
    Return posterior standard deviation for a parameter key.
    Returns np.nan if the key is not available.
    """
    if key in posterior_samples:
        values = np.asarray(posterior_samples[key], dtype=float)
        return np.nanstd(values)

    return np.nan


def get_slope_sigma_kms(slope_key, posterior_samples):
    """
    Get the slope uncertainty in km/s.

    First tries the instrument-specific slope key.
    If it does not exist, falls back to the shared/master slope key.
    """
    sigma = get_posterior_sigma(slope_key, posterior_samples)

    if np.isfinite(sigma):
        return sigma, slope_key

    sigma_shared = get_posterior_sigma(SHARED_RV_SLOPE_KEY, posterior_samples)

    if np.isfinite(sigma_shared):
        return sigma_shared, SHARED_RV_SLOPE_KEY

    return np.nan, None


def rv_baseline(alles, posterior_params_median, inst, time):
    """
    Return the fitted allesfitter RV baseline in m/s.

    allesfitter RV quantities are assumed to be in km/s, so we multiply by RV_SCALE.
    """
    data_time = np.asarray(alles.data[inst]["time"], dtype=float)
    time = np.asarray(time, dtype=float)

    mode = alles.settings[f"baseline_rv_{inst}"]
    offset = posterior_params_median[f"baseline_offset_rv_{inst}"] * RV_SCALE

    if mode == "sample_offset":
        return offset * np.ones_like(time, dtype=float)

    if mode == "sample_linear":
        slope = posterior_params_median[f"baseline_slope_rv_{inst}"] * RV_SCALE
        time_norm = (time - data_time[0]) / (data_time[-1] - data_time[0])
        return offset + slope * time_norm

    raise ValueError(f"Unsupported RV baseline mode for {inst}: {mode}")


def rv_baseline_no_offset(alles, posterior_params_median, inst, time):
    """
    Return only the fitted linear part of the RV baseline in m/s, excluding the offset.
    """
    data_time = np.asarray(alles.data[inst]["time"], dtype=float)
    time = np.asarray(time, dtype=float)

    mode = alles.settings[f"baseline_rv_{inst}"]

    if mode == "sample_offset":
        return np.zeros_like(time, dtype=float)

    if mode == "sample_linear":
        slope = posterior_params_median[f"baseline_slope_rv_{inst}"] * RV_SCALE
        time_norm = (time - data_time[0]) / (data_time[-1] - data_time[0])
        return slope * time_norm

    raise ValueError(f"Unsupported RV baseline mode for {inst}: {mode}")


def rv_offset(posterior_params_median, inst):
    """
    Return fitted RV offset in m/s.
    """
    return posterior_params_median[f"baseline_offset_rv_{inst}"] * RV_SCALE


def planet_model(alles, inst, time):
    """
    Return posterior median planetary RV model in m/s.
    """
    return alles.get_posterior_median_model(inst=inst, key="rv", xx=time) * RV_SCALE


def to_phase(period, epoch, time):
    return ((time - epoch) / period) % 1


def save_time_plot(alles, posterior_params_median, dirname):
    all_times = np.hstack([alles.data[inst]["time"] for inst in RV_INSTRUMENTS])
    t0 = int(all_times.min())
    time_grid = np.linspace(all_times.min() - 5, all_times.max() + 5, 2500)

    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        height_ratios=[3, 1],
        dpi=120,
        figsize=(7, 5)
    )

    # Representative red model line using HARPS:
    # planet model + fitted HARPS baseline, with HARPS offset removed.
    harps_offset = rv_offset(posterior_params_median, "HARPS")
    harps_model_grid = (
        planet_model(alles, "HARPS", time_grid)
        + rv_baseline(alles, posterior_params_median, "HARPS", time_grid)
        - harps_offset
    )

    ax[0].plot(
        time_grid - t0,
        harps_model_grid,
        "r-",
        lw=1.4,
        label="Planet + fitted RV baseline"
    )

    for inst in RV_INSTRUMENTS:
        data = alles.data[inst]

        time = np.asarray(data["time"], dtype=float)
        rv = np.asarray(data["rv"], dtype=float) * RV_SCALE
        rv_err = np.asarray(data["rv_err"], dtype=float) * RV_SCALE

        offset = rv_offset(posterior_params_median, inst)

        full_model = (
            planet_model(alles, inst, time)
            + rv_baseline(alles, posterior_params_median, inst, time)
        )

        # Time plot:
        # subtract only the instrumental offset from the data,
        # so the fitted linear baseline remains visible.
        ax[0].errorbar(
            time - t0,
            rv - offset,
            yerr=rv_err,
            fmt=MARKERS[inst],
            capsize=2,
            color=COLORS[inst],
            label=DISPLAY_NAMES[inst],
        )

        # Residuals:
        # subtract the full fitted model: planet + offset + slope.
        ax[1].errorbar(
            time - t0,
            rv - full_model,
            yerr=rv_err,
            fmt=MARKERS[inst],
            capsize=2,
            color=COLORS[inst],
        )

    ax[1].axhline(0, color="r", lw=1)

    # ax[0].legend(loc="best", fontsize=9)
    ax[0].set_ylabel("RV - offset (m s$^{-1}$)")
    ax[1].set_ylabel("O-C (m s$^{-1}$)")
    ax[1].set_xlabel(f"Time [BJD - {t0}]")

    fig.tight_layout()

    pdf_path = os.path.join(dirname, "TIC_RV_TIME.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return pdf_path


def save_phase_plot(alles, posterior_params_median, period, epoch, dirname):
    phase_grid = np.linspace(0, 1, 2000)
    time_grid = epoch + phase_grid * period

    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        height_ratios=[3, 1],
        dpi=120,
        figsize=(7, 5)
    )

    # Phase-folded model: only the Keplerian planet model.
    # The fitted RV baselines are removed from the data.
    ax[0].plot(
        phase_grid,
        planet_model(alles, "HARPS", time_grid),
        "r-",
        lw=1.4,
        label="Planet model"
    )

    for inst in RV_INSTRUMENTS:
        data = alles.data[inst]

        time = np.asarray(data["time"], dtype=float)
        rv = np.asarray(data["rv"], dtype=float) * RV_SCALE
        rv_err = np.asarray(data["rv_err"], dtype=float) * RV_SCALE

        phase = to_phase(period, epoch, time)

        baseline = rv_baseline(alles, posterior_params_median, inst, time)
        model = planet_model(alles, inst, time)

        # Remove the full fitted baseline before phase-folding.
        ax[0].errorbar(
            phase,
            rv - baseline,
            yerr=rv_err,
            fmt=MARKERS[inst],
            capsize=2,
            color=COLORS[inst],
            label=DISPLAY_NAMES[inst],
        )

        # Residuals after removing baseline and planet model.
        ax[1].errorbar(
            phase,
            rv - baseline - model,
            yerr=rv_err,
            fmt=MARKERS[inst],
            capsize=2,
            color=COLORS[inst],
        )

    ax[1].axhline(0, color="r", lw=1)

    # ax[0].legend(loc="best", fontsize=9)
    ax[0].set_ylabel("RV - baseline (m s$^{-1}$)")
    ax[1].set_ylabel("O-C (m s$^{-1}$)")
    ax[1].set_xlabel("Orbital Phase")

    fig.tight_layout()

    pdf_path = os.path.join(dirname, "TIC_RV_PHASE.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return pdf_path


def print_available_slope_keys(posterior_params_median, posterior_samples):
    print("\nAvailable posterior sample keys containing 'slope':")
    sample_slope_keys = [
        key for key in posterior_samples.keys()
        if "slope" in key
    ]

    if len(sample_slope_keys) == 0:
        print("  No slope keys found in posterior_samples.")
    else:
        for key in sample_slope_keys:
            print(f"  {key}")

    print("\nAvailable posterior median keys containing 'slope':")
    median_slope_keys = [
        key for key in posterior_params_median.keys()
        if "slope" in key
    ]

    if len(median_slope_keys) == 0:
        print("  No slope keys found in posterior_params_median.")
    else:
        for key in median_slope_keys:
            print(f"  {key}")


def main():
    sns.set(
        context="paper",
        style="ticks",
        palette="deep",
        font_scale=1.8,
        color_codes=True
    )
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
    sns.set_context(rc={"lines.markeredgewidth": 1})

    plot_images()

    alles = allesfitter.allesclass(DATA_DIR)

    posterior_params_median = alles.posterior_params_median
    posterior_samples = alles.posterior_params

    print_available_slope_keys(
        posterior_params_median,
        posterior_samples
    )

    period = posterior_params_median["b_period"]
    epoch = posterior_params_median["b_epoch"]

    # Recalculate RV uncertainties including fitted jitter.
    # These remain in native allesfitter units, assumed to be km/s.
    # They are converted to m/s only when plotting/printing.
    for inst in RV_INSTRUMENTS:
        data = alles.data[inst]
        data["rv_err"] = np.sqrt(
            data["white_noise_rv"] ** 2
            + get_rv_jitter(inst, posterior_params_median) ** 2
        )

    print_fitted_rv_baselines(
        alles,
        posterior_params_median,
        posterior_samples
    )

    output_paths = []
    output_paths.append(
        save_time_plot(
            alles,
            posterior_params_median,
            DATA_DIR
        )
    )
    output_paths.append(
        save_phase_plot(
            alles,
            posterior_params_median,
            period,
            epoch,
            DATA_DIR
        )
    )

    print("\nSaved plots:")
    for path in output_paths:
        print(f"  {path}")


def get_global_rv_time_baseline(alles):
    """
    Return the full RV time baseline using all RV instruments together.

    This is measured from the first RV observation to the last RV observation
    across HARPS, CORALIE_1, and CORALIE_2 combined.
    """
    all_times = np.hstack([
        np.asarray(alles.data[inst]["time"], dtype=float)
        for inst in RV_INSTRUMENTS
    ])

    t_min = np.nanmin(all_times)
    t_max = np.nanmax(all_times)
    dt_global = t_max - t_min

    return t_min, t_max, dt_global


def print_fitted_rv_baselines(alles, posterior_params_median, posterior_samples):
    print("\nFitted RV baselines extracted directly from allesfitter:")
    print("-" * 80)

    # Global RV baseline using all RV instruments together
    global_t_min, global_t_max, global_dt = get_global_rv_time_baseline(alles)

    print("\nGlobal RV time baseline from all instruments combined:")
    print(f"  first RV BJD                    : {global_t_min:.6f}")
    print(f"  last RV BJD                     : {global_t_max:.6f}")
    print(f"  global RV baseline              : {global_dt:.6f} days")
    print(f"  global RV baseline              : {global_dt / 365.25:.6f} years")
    print("-" * 80)

    for inst in RV_INSTRUMENTS:
        mode = alles.settings[f"baseline_rv_{inst}"]

        offset_key = f"baseline_offset_rv_{inst}"
        slope_key = f"baseline_slope_rv_{inst}"

        offset_kms = posterior_params_median[offset_key]
        offset_ms = offset_kms * RV_SCALE

        print(f"{DISPLAY_NAMES[inst]}")
        print(f"  baseline mode                  : {mode}")
        print(f"  offset parameter               : {offset_key}")
        print(f"  offset                         : {offset_ms:.6f} m/s")
        print(f"  offset                         : {offset_kms:.9f} km/s")

        if mode == "sample_linear":
            slope_kms = posterior_params_median[slope_key]
            slope_ms = slope_kms * RV_SCALE

            # Instrument-specific baseline
            time = np.asarray(alles.data[inst]["time"], dtype=float)
            dt_inst = time[-1] - time[0]

            # Per-day/year using instrument-only baseline
            slope_per_day_inst_ms = slope_ms / dt_inst
            slope_per_year_inst_ms = slope_per_day_inst_ms * 365.25

            slope_per_day_inst_kms = slope_kms / dt_inst
            slope_per_year_inst_kms = slope_per_day_inst_kms * 365.25

            # Per-day/year using global RV baseline
            slope_per_day_global_ms = slope_ms / global_dt
            slope_per_year_global_ms = slope_per_day_global_ms * 365.25

            slope_per_day_global_kms = slope_kms / global_dt
            slope_per_year_global_kms = slope_per_day_global_kms * 365.25

            slope_sigma_kms, sigma_source_key = get_slope_sigma_kms(
                slope_key,
                posterior_samples
            )
            slope_sigma_ms = slope_sigma_kms * RV_SCALE

            if np.isfinite(slope_sigma_kms):
                # Uncertainty using instrument-only baseline
                slope_per_day_sigma_inst_ms = slope_sigma_ms / dt_inst
                slope_per_year_sigma_inst_ms = slope_per_day_sigma_inst_ms * 365.25

                slope_per_day_sigma_inst_kms = slope_sigma_kms / dt_inst
                slope_per_year_sigma_inst_kms = slope_per_day_sigma_inst_kms * 365.25

                # Uncertainty using global RV baseline
                slope_per_day_sigma_global_ms = slope_sigma_ms / global_dt
                slope_per_year_sigma_global_ms = slope_per_day_sigma_global_ms * 365.25

                slope_per_day_sigma_global_kms = slope_sigma_kms / global_dt
                slope_per_year_sigma_global_kms = slope_per_day_sigma_global_kms * 365.25
            else:
                slope_per_day_sigma_inst_ms = np.nan
                slope_per_year_sigma_inst_ms = np.nan
                slope_per_day_sigma_inst_kms = np.nan
                slope_per_year_sigma_inst_kms = np.nan

                slope_per_day_sigma_global_ms = np.nan
                slope_per_year_sigma_global_ms = np.nan
                slope_per_day_sigma_global_kms = np.nan
                slope_per_year_sigma_global_kms = np.nan

            print(f"  slope parameter                : {slope_key}")
            print(f"  slope key in posterior medians : {slope_key in posterior_params_median}")
            print(f"  slope key in posterior samples : {slope_key in posterior_samples}")

            if sigma_source_key is not None:
                print(f"  uncertainty source key         : {sigma_source_key}")
            else:
                print("  uncertainty source key         : None found")

            print(f"  slope                          : {slope_ms:.6f} m/s over fitted baseline")
            print(f"  slope                          : {slope_kms:.9f} km/s over fitted baseline")

            print("\n  Instrument-only time baseline:")
            print(f"    time baseline                : {dt_inst:.6f} days")
            print(f"    time baseline                : {dt_inst / 365.25:.6f} years")
            print(f"    slope per day                : {slope_per_day_inst_ms:.6f} m/s/day")
            print(f"    slope per year               : {slope_per_year_inst_ms:.6f} m/s/year")
            print(f"    slope per day                : {slope_per_day_inst_kms:.9f} km/s/day")
            print(f"    slope per year               : {slope_per_year_inst_kms:.9f} km/s/year")
            print(f"    slope uncertainty            : {slope_sigma_ms:.6f} m/s over fitted baseline")
            print(f"    slope uncertainty            : {slope_sigma_kms:.9f} km/s over fitted baseline")
            print(f"    slope-per-day uncertainty    : {slope_per_day_sigma_inst_ms:.6f} m/s/day")
            print(f"    slope-per-year uncertainty   : {slope_per_year_sigma_inst_ms:.6f} m/s/year")
            print(f"    slope-per-day uncertainty    : {slope_per_day_sigma_inst_kms:.9f} km/s/day")
            print(f"    slope-per-year uncertainty   : {slope_per_year_sigma_inst_kms:.9f} km/s/year")

            print("\n  Global RV time baseline:")
            print(f"    time baseline                : {global_dt:.6f} days")
            print(f"    time baseline                : {global_dt / 365.25:.6f} years")
            print(f"    slope per day                : {slope_per_day_global_ms:.6f} m/s/day")
            print(f"    slope per year               : {slope_per_year_global_ms:.6f} m/s/year")
            print(f"    slope per day                : {slope_per_day_global_kms:.9f} km/s/day")
            print(f"    slope per year               : {slope_per_year_global_kms:.9f} km/s/year")
            print(f"    slope uncertainty            : {slope_sigma_ms:.6f} m/s over fitted baseline")
            print(f"    slope uncertainty            : {slope_sigma_kms:.9f} km/s over fitted baseline")
            print(f"    slope-per-day uncertainty    : {slope_per_day_sigma_global_ms:.6f} m/s/day")
            print(f"    slope-per-year uncertainty   : {slope_per_year_sigma_global_ms:.6f} m/s/year")
            print(f"    slope-per-day uncertainty    : {slope_per_day_sigma_global_kms:.9f} km/s/day")
            print(f"    slope-per-year uncertainty   : {slope_per_year_sigma_global_kms:.9f} km/s/year")

        print("-" * 80)

if __name__ == "__main__":
    main()