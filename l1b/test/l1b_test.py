from common.io.writeToa import readToa
from config.globalConfig import globalConfig

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import contextlib
import io

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

reference = r"C:\Users\Tania\Documents\GitHub\EODP\EODP_TER_2021\EODP-TS-L1B\output_Lucia"
outdir_eq = r"C:\Users\Tania\Documents\GitHub\EODP\EODP_TER_2021\EODP-TS-L1B\output_Tania_eq"
outdir_noteq = r"C:\Users\Tania\Documents\GitHub\EODP\EODP_TER_2021\EODP-TS-L1B\output_Tania_noteq"
inputdir = r"C:\Users\Tania\Documents\GitHub\EODP\EODP_TER_2021\EODP-TS-L1B\input"

# logger
logger = logging.getLogger("L1B_test")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(os.path.join(logs_dir, "l1b_test.log"), mode="w")
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

rel_tol_percent = 1e-3    # maximum tolerance

def read_toa_silent(directory, filename):
    with contextlib.redirect_stdout(io.StringIO()):
        toa = readToa(directory, filename)
    return toa

def compare_output(toa_tania, toa_lucia, filename):
    # check dimensions
    if toa_tania.shape != toa_lucia.shape:
        logger.info(filename + ": FAIL - Different dimensions " + str(toa_tania.shape) + " vs " + str(toa_lucia.shape))
        return False

    # absolute difference
    abs_diff = np.abs(toa_tania - toa_lucia)

    # pixels where relative difference can be calculated
    valid = toa_lucia != 0

    # relative difference in %
    rel_diff = np.zeros_like(toa_tania, dtype=float)
    rel_diff[valid] = (abs_diff[valid] / np.abs(toa_lucia[valid]) * 100)

    # maximum relative difference
    max_rel_diff = np.max(rel_diff[valid])

    # pass/fail
    passed = np.all(rel_diff[valid] < rel_tol_percent)

    if passed:
        logger.info(filename + ": PASS | Max relative difference = " + f"{max_rel_diff:.3e} %")
    else:
        logger.info(filename + ": FAIL | Max relative difference = " + f"{max_rel_diff:.3e} %")
    return passed

logger.info("---------------------------------------------------------------------")
logger.info("L1B CROSS-VALIDATION")

logger.info("---------------------------------------------------------------------")
logger.info("TEST 1 - EQUALIZATION")
logger.info("---------------------------------------------------------------------")

for band in bands:
    filename = "l1b_toa_eq_" + band + ".nc"
    toa_tania = read_toa_silent(outdir_eq, filename)
    toa_lucia = read_toa_silent(reference, filename)
    compare_output(toa_tania, toa_lucia, filename)

logger.info("---------------------------------------------------------------------")
logger.info("TEST 2 - RESTORATION")
logger.info("---------------------------------------------------------------------")

for band in bands:
    filename = "l1b_toa_" + band + ".nc"
    toa_tania = read_toa_silent(outdir_eq, filename)
    toa_lucia = read_toa_silent(reference, filename)
    compare_output(toa_tania, toa_lucia, filename)

logger.info("---------------------------------------------------------------------")
logger.info("TEST 3 - EFFECT OF EQUALIZATION")
logger.info("---------------------------------------------------------------------")

for band in bands:
    true_signal = read_toa_silent(inputdir, "ism_toa_isrf_" + band + ".nc")
    toa_eq = read_toa_silent(outdir_eq, "l1b_toa_" + band + ".nc")
    toa_noteq = read_toa_silent(outdir_noteq, "l1b_toa_" + band + ".nc")

    mid_alt = int(true_signal.shape[0] / 2)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(true_signal[mid_alt, :], color="blue", label="True signal")
    ax.plot(toa_noteq[mid_alt, :], color="red", label="L1B without equalization")
    ax.plot(toa_eq[mid_alt, :], color="black", label="L1B with equalization")
    ax.set_title("Effect of Equalization for " + band)
    ax.set_xlabel("ACT pixel [-]")
    ax.set_ylabel("TOA [mW/m2/sr]")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    figure_name = "effect_equalization_" + band + ".png"
    fig.savefig(os.path.join(figures_dir, figure_name), dpi=300, bbox_inches="tight")
    logger.info(band + ": Figure generated - " + figure_name)
    plt.close(fig)