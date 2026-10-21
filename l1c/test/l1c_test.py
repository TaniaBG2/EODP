from common.io.l1cProduct import readL1c
from common.io.readGeodetic import readGeodetic
from config.globalConfig import globalConfig

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import contextlib
import io
from scipy.spatial import cKDTree


# configuration
myglobal = globalConfig()
bands = myglobal.bands

# test directories
test_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(test_dir, "figures")
logs_dir = os.path.join(test_dir, "logs")
outputs_dir = os.path.join(test_dir, "outputs")

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

# input and output directories
project_dir = os.path.dirname(os.path.dirname(test_dir))
test_data_dir = os.path.join(project_dir, "EODP_TER_2021")

reference = os.path.join(test_data_dir, "EODP-TS-L1C", "output_Lucia")
outdir = os.path.join(test_data_dir, "EODP-TS-L1C", "output")
inputdir = os.path.join(test_data_dir, "EODP-TS-L1C", "input", "gm_alt100_act_150")

# logger
logger = logging.getLogger("L1C_test")
logger.setLevel(logging.INFO)
logger.propagate = False

for handler in logger.handlers[:]:
    logger.removeHandler(handler)
    handler.close()

file_handler = logging.FileHandler(os.path.join(logs_dir, "l1c_test.log"), mode="w")
console_handler = logging.StreamHandler()

formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# tolerances and constants
rel_tol_percent = 1e-3
coord_tol_deg = 1e-6
R = 6371000  # Earth radius [m]


def read_l1c_silent(directory, filename):
    with contextlib.redirect_stdout(io.StringIO()):
        toa, lat, lon = readL1c(directory, filename)
    return np.asarray(toa).ravel(), np.asarray(lat).ravel(), np.asarray(lon).ravel()


def read_geodetic_silent(directory, filename):
    with contextlib.redirect_stdout(io.StringIO()):
        lat, lon = readGeodetic(directory, filename)
    return np.asarray(lat), np.asarray(lon)


def compare_output(toa_tania, lat_tania, lon_tania,
                   toa_ref, lat_ref, lon_ref, filename):

    if not (len(toa_tania) == len(lat_tania) == len(lon_tania) == len(toa_ref) == len(lat_ref) == len(lon_ref)):
        logger.info(filename + ": FAIL - Different dimensions")
        return False

    if len(toa_tania) == 0:
        logger.info(filename + ": FAIL - Empty output")
        return False

    arrays = (toa_tania, lat_tania, lon_tania, toa_ref, lat_ref, lon_ref)

    if not all(np.all(np.isfinite(a)) for a in arrays):
        logger.info(filename + ": FAIL - Non-finite values")
        return False

    coords_tania = np.column_stack((lat_tania, lon_tania))
    coords_ref = np.column_stack((lat_ref, lon_ref))

    keys_tania = [tuple(p) for p in np.round(coords_tania, 6)]
    keys_ref = [tuple(p) for p in np.round(coords_ref, 6)]

    if (
        len(set(keys_tania)) != len(keys_tania)
        or len(set(keys_ref)) != len(keys_ref)
        or set(keys_tania) != set(keys_ref)
    ):
        logger.info(filename + ": FAIL - Different geographical coordinates")
        return False

    reference_index = {key: i for i, key in enumerate(keys_ref)}
    order = np.array([reference_index[key] for key in keys_tania])

    toa_ref = toa_ref[order]
    lat_ref = lat_ref[order]
    lon_ref = lon_ref[order]

    max_coord_diff = max(np.max(np.abs(lat_tania - lat_ref)), np.max(np.abs(lon_tania - lon_ref)))
    abs_diff = np.abs(toa_tania - toa_ref)
    valid = toa_ref != 0
    rel_diff = np.zeros_like(toa_tania, dtype=float)
    rel_diff[valid] = (abs_diff[valid] / np.abs(toa_ref[valid]) * 100)
    max_rel_diff = (np.max(rel_diff[valid]) if np.any(valid) else 0.0)

    passed = (
        np.all(np.isfinite(rel_diff))
        and np.all(rel_diff[valid] < rel_tol_percent)
        and np.all(toa_tania[~valid] == 0)
        and max_coord_diff < coord_tol_deg
    )

    logger.info(
        filename + ": " + ("PASS" if passed else "FAIL")
        + " | Max relative difference = " + f"{max_rel_diff:.3e} %"
        + " | Max coordinate difference = " + f"{max_coord_diff:.3e} deg"
    )

    return bool(passed)


def haversine(lat1, lon1, lat2, lon2):

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2)

    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def save_figure(fig, filename):

    fig.savefig(os.path.join(figures_dir, filename), dpi=300, bbox_inches="tight")

    plt.close(fig)
    logger.info("Figure generated - " + filename)


def geographic_aspect(lat):

    return 1 / np.cos(np.radians(np.mean(lat)))


# ---------------------------------------------------------------------
# TEST 1 - MGRS ORTHORECTIFICATION
# ---------------------------------------------------------------------

logger.info("---------------------------------------------------------------------")
logger.info("L1C CROSS-VALIDATION")
logger.info("---------------------------------------------------------------------")
logger.info("TEST 1 - MGRS ORTHORECTIFICATION")
logger.info("---------------------------------------------------------------------")

results = {}
products = {}

for band in bands:

    filename = "l1c_toa_" + band + ".nc"
    toa_tania, lat_tania, lon_tania = read_l1c_silent(outdir, filename)
    toa_ref, lat_ref, lon_ref = read_l1c_silent(reference, filename)
    results[band] = compare_output(toa_tania, lat_tania, lon_tania, toa_ref, lat_ref, lon_ref, filename)
    products[band] = (toa_tania, lat_tania, lon_tania)

overall = all(results.values())

logger.info("OVERALL CROSS-VALIDATION: " + ("PASS" if overall else "FAIL"))


# ---------------------------------------------------------------------
# TEST 2 - L1B AND L1C FOOTPRINT
# ---------------------------------------------------------------------

logger.info("---------------------------------------------------------------------")
logger.info("TEST 2 - L1B AND L1C FOOTPRINT")
logger.info("---------------------------------------------------------------------")

lat_l1b, lon_l1b = read_geodetic_silent(inputdir, "geolocation.nc")

perimeter_lat = np.concatenate((
    lat_l1b[0, :],
    lat_l1b[1:, -1],
    lat_l1b[-1, -2::-1],
    lat_l1b[-2:0:-1, 0],
    lat_l1b[0, :1]
))

perimeter_lon = np.concatenate((
    lon_l1b[0, :],
    lon_l1b[1:, -1],
    lon_l1b[-1, -2::-1],
    lon_l1b[-2:0:-1, 0],
    lon_l1b[0, :1]
))


first_band = bands[0]
_, lat_l1c, lon_l1c = products[first_band]

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(perimeter_lon, perimeter_lat, color="black", linewidth=1.5, label="L1B")

ax.scatter(lon_l1c, lat_l1c, s=4, color="red", label="L1C MGRS", rasterized=True)
ax.set_title("Projection on ground - L1B vs L1C")
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")
ax.set_aspect(geographic_aspect(lat_l1b), adjustable="datalim")
ax.grid(True, alpha=0.4)
ax.legend(loc="upper right")
plt.tight_layout()
save_figure(fig, "footprint_L1B_L1C.png")


for band in bands:

    toa_l1c, lat_l1c_band, lon_l1c_band = products[band]
    fig, ax = plt.subplots(figsize=(12, 7))
    sc = ax.scatter(lon_l1c_band, lat_l1c_band, c=toa_l1c, cmap="viridis", s=9, linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, label="TOA [mW/m²/sr]")
    ax.set_title("L1C TOA for " + band)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_aspect(geographic_aspect(lat_l1c_band), adjustable="datalim")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    figure_name = "l1c_toa_" + band + ".png"
    save_figure(fig, figure_name)

    logger.info(
        band
        + ": TOA min = " + f"{np.min(toa_l1c):.3f}"
        + " | max = " + f"{np.max(toa_l1c):.3f}"
        + " | mean = " + f"{np.mean(toa_l1c):.3f}"
        + " | negative samples = " + str(np.count_nonzero(toa_l1c < 0))
    )


# ---------------------------------------------------------------------
# TEST 3 - GROUND SAMPLING DISTANCE
# ---------------------------------------------------------------------

logger.info("---------------------------------------------------------------------")
logger.info("TEST 3 - GROUND SAMPLING DISTANCE")
logger.info("---------------------------------------------------------------------")

gsd_alt = haversine(lat_l1b[:-1, :], lon_l1b[:-1, :], lat_l1b[1:, :], lon_l1b[1:, :]).ravel()
gsd_act = haversine( lat_l1b[:, :-1], lon_l1b[:, :-1], lat_l1b[:, 1:], lon_l1b[:, 1:]).ravel()


lat0 = np.radians(np.mean(lat_l1c))

x = R * np.radians(lon_l1c) * np.cos(lat0)
y = R * np.radians(lat_l1c)

points = np.column_stack((x, y))

tree = cKDTree(points)
distances, indices = tree.query(points, k=2)

neighbours = indices[:, 1]

gsd_l1c = haversine(lat_l1c, lon_l1c, lat_l1c[neighbours], lon_l1c[neighbours])

logger.info("L1B ALT mean GSD = " + f"{np.mean(gsd_alt):.3f} m")
logger.info("L1B ACT mean GSD = " + f"{np.mean(gsd_act):.3f} m")
logger.info("L1C mean nearest-neighbour distance = " + f"{np.mean(gsd_l1c):.3f} m")

logger.info("L1B ALT standard deviation = " + f"{np.std(gsd_alt):.3f} m")
logger.info("L1B ACT standard deviation = " + f"{np.std(gsd_act):.3f} m")
logger.info("L1C spacing standard deviation = " + f"{np.std(gsd_l1c):.3f} m")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(gsd_alt, bins=35, alpha=0.6, label="L1B ALT")
ax1.hist(gsd_act, bins=35, alpha=0.6, label="L1B ACT")
ax1.axvline(np.mean(gsd_alt), color="blue", linestyle="--", label=f"ALT mean = {np.mean(gsd_alt):.3f} m")
ax1.axvline(np.mean(gsd_act), color="orange", linestyle="--", label=f"ACT mean = {np.mean(gsd_act):.3f} m")
ax1.set_title("L1B Ground Sampling Distance")
ax1.set_xlabel("Distance [m]")
ax1.set_ylabel("Number of samples")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper center", fontsize=8)

ax2.hist(gsd_l1c, bins=35, alpha=0.7, color="tab:green", label="Nearest neighbour")
ax2.axvline(np.mean(gsd_l1c), color="black", linestyle="--", label=f"Mean = {np.mean(gsd_l1c):.3f} m")
ax2.set_title("L1C Ground Sampling Distance")
ax2.set_xlabel("Distance [m]")
ax2.set_ylabel("Number of samples")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right", fontsize=8)

plt.tight_layout()

save_figure(fig, "gsd_distribution.png")
