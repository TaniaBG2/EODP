import numpy as np
import logging

from common.io.writeToa import readToa

# configuration
tania_dir = r"C:\\Users\\Tania\\Documents\\GitHub\\EODP\\EODP_TER_2021\\EODP-TS-ISM\\output_Tania"
lucia_dir = r"C:\\Users\\Tania\\Documents\\GitHub\\EODP\\EODP_TER_2021\\EODP-TS-ISM\\output_Lucia"

rel_tol_percent = 1e-3

# logger
logger = logging.getLogger("ISM_test")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/test_ism.log", mode="w")

console_handler = logging.StreamHandler()

formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def compare_output(tania, lucia, name):

    # check dimensions
    if tania.shape != lucia.shape:
        logger.info(f"{name}: FAIL - Different dimensions "f"{tania.shape} vs {lucia.shape}")
        return False

    # absolute difference
    abs_diff = np.abs(tania - lucia)

    # relative difference (%)
    valid = lucia != 0

    rel_diff = np.zeros_like(tania, dtype=float)

    rel_diff[valid] = (abs_diff[valid] / np.abs(lucia[valid]) * 100)

    max_rel_diff = np.max(rel_diff[valid])

    passed = np.all(rel_diff[valid] < REL_TOL_PERCENT)

    result = "PASS" if passed else "FAIL"

    logger.info(f"{name}: {result} | "f"Max relative difference = {max_rel_diff:.3e} %")

    return passed

# spectral integration
bands = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]

logger.info("========================================")
logger.info("ISM CROSS-VALIDATION")
logger.info("========================================")
logger.info("")

logger.info("OPTICAL STAGE")
logger.info("----------------------------------------")

for band in bands:
    filename = f"ism_toa_isrf_{band}.nc"

    tania = readToa(tania_dir, filename)
    lucia = readToa(lucia_dir, filename)

    compare_output(tania, lucia, filename)

# optical outputs
optical_outputs = ["ism_toa_isrf","ism_toa_optical"]

# detection outputs
detection_outputs = ["ism_toa_e","ism_toa_ds","ism_toa_prnu","ism_toa_detection"]