
from common.io.writeToa import readToa
from common.io.readMat import readMat
from config.globalConfig import globalConfig

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import io
import contextlib

# Configuration
myglobal = globalConfig()
bands = myglobal.bands
rel_tol_percent = 1e-3

# Test directories
test_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(test_dir, 'figures')
logs_dir = os.path.join(test_dir, 'logs')
outputs_dir = os.path.join(test_dir, 'outputs')
for directory in (figures_dir, logs_dir, outputs_dir):
    os.makedirs(directory, exist_ok=True)

project_dir = os.path.dirname(os.path.dirname(test_dir))
test_data_dir = os.path.join(project_dir, 'EODP_TER_2021', 'EODP-TS-ISM')
reference = os.path.join(test_data_dir, 'output_Lucia')
outdir = os.path.join(test_data_dir, 'output_Tania')

# Logger
logger = logging.getLogger('ISM_test')
logger.setLevel(logging.INFO)
logger.propagate = False
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
    handler.close()
formatter = logging.Formatter('%(message)s')
for handler in (logging.FileHandler(os.path.join(logs_dir, 'ism_test.log'), mode='w'),
                logging.StreamHandler()):
    handler.setFormatter(formatter)
    logger.addHandler(handler)

results = {}


def section(title):
    logger.info('-' * 69)
    logger.info(title)
    logger.info('-' * 69)


def read_toa_silent(directory, filename):
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        logger.info(filename + ': MISSING - ' + path)
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        if filename.startswith("H"):
            return np.asarray(readMat(directory, filename))
        return np.asarray(readToa(directory, filename))


def compare_output(output, ref, filename):
    if output is None or ref is None:
        results[filename] = 'MISSING'
        return
    if output.shape != ref.shape or output.size == 0:
        logger.info(filename + ': FAIL - Invalid or different dimensions')
        results[filename] = 'FAIL'
        return
    if not np.all(np.isfinite(output)) or not np.all(np.isfinite(ref)):
        logger.info(filename + ': FAIL - Non-finite values')
        results[filename] = 'FAIL'
        return
    diff = np.abs(output.astype(float) - ref.astype(float))
    valid = ref != 0
    max_rel = float(np.max(diff[valid] / np.abs(ref[valid]) * 100)) if np.any(valid) else 0.0
    passed = (np.all(diff[valid] / np.abs(ref[valid]) * 100 <= rel_tol_percent)
              and np.all(diff[~valid] == 0))
    status = 'PASS' if passed else 'FAIL'
    logger.info(f'{filename}: {status} | Max relative difference = {max_rel:.3e} %'
                f' | Max absolute difference = {np.max(diff):.3e}')
    results[filename] = status


section('ISM CROSS-VALIDATION')

# TEST 1 - Optical stage
section('TEST 1 - OPTICAL STAGE')
for output in ('ism_toa_isrf', 'ism_toa_optical'):
    for band in bands:
        filename = f'{output}_{band}.nc'
        compare_output(read_toa_silent(outdir, filename),
                       read_toa_silent(reference, filename), filename)

# TEST 2 - Detection and video chain
# A deterministic seed is configured in ismConfig. Compare the actual arrays,
# rather than automatically exempting the random stages from cross-validation.
section('TEST 2 - DETECTION AND VIDEO CHAIN')
for output in ('ism_toa_e', 'ism_toa_prnu', 'ism_toa_ds',
               'ism_toa_detection', 'ism_toa'):
    for band in bands:
        filename = f'{output}_{band}.nc'
        own = read_toa_silent(outdir, filename)
        ref = read_toa_silent(reference, filename)
        compare_output(own, ref, filename)
        if output == 'ism_toa' and own is not None:
            max_dn = 2**12 - 1
            logger.info(f'  Digital range {band}: min={np.min(own):.0f}, '
                        f'max={np.max(own):.0f}, pixels above {max_dn}='
                        f'{np.count_nonzero(own > max_dn)}')

# TEST 3 - System MTF and six contributions
section('TEST 3 - SYSTEM MTF AND ITS SIX CONTRIBUTORS')
mtf_names = ('Hdiff', 'Hdefoc', 'Hwfe', 'Hdet', 'Hsmear', 'Hmotion', 'Hsys')
mtf_labels = ('Diffraction MTF', 'Defocus MTF', 'WFE aberrations MTF',
              'Detector MTF', 'Smearing MTF', 'Motion blur MTF', 'System MTF')

for band in bands:
    arrays = {}
    for name in mtf_names:
        arrays[name] = read_toa_silent(reference, f'{name}_{band}.nc')
    if any(value is None for value in arrays.values()):
        logger.info(f'{band}: MISSING - MTF reference matrices')
        results[f'MTF_{band}'] = 'MISSING'
        continue

    shape = arrays['Hsys'].shape
    if any(value.shape != shape for value in arrays.values()):
        logger.info(f'{band}: FAIL - Inconsistent MTF dimensions')
        results[f'MTF_{band}'] = 'FAIL'
        continue

    nlines, ncolumns = shape
    half_alt, half_act = nlines // 2, ncolumns // 2
    product = np.ones(shape, dtype=float)
    for name in mtf_names[:-1]:
        product *= arrays[name]
    compare_output(product, arrays['Hsys'], f'MTF_product_{band}')

    # Nyquist lies at the first sample (-0.5) on FFT-shifted even grids.
    for direction, values in (
            ('ACT', {name: float(arrays[name][half_alt, 0]) for name in mtf_names}),
            ('ALT', {name: float(arrays[name][0, half_act]) for name in mtf_names})
    ):
        dominant = sorted(mtf_names[:-1], key=lambda name: values[name])[:2]
        logger.info(f'{band} {direction}: System MTF at Nyquist = {values["Hsys"]:.6f}'
                    f' | Two strongest attenuations: '
                    f'{mtf_labels[mtf_names.index(dominant[0])]} '
                    f'({values[dominant[0]]:.6f}), '
                    f'{mtf_labels[mtf_names.index(dominant[1])]} '
                    f'({values[dominant[1]]:.6f})')

        # Arrays are FFT-shifted: zero frequency is at the centre.
        # The positive-frequency half runs from the centre to the final
        # column/row. Nyquist (-0.5) is stored at index 0 for even sizes.
        # Re-label that equivalent endpoint as +0.5 for plotting.
        if direction == 'ACT':
            indices = np.r_[half_act:ncolumns, 0]
            frequencies = np.r_[np.arange(ncolumns - half_act) / ncolumns, 0.5]
            extract = lambda arr: arr[half_alt, indices]
        else:
            indices = np.r_[half_alt:nlines, 0]
            frequencies = np.r_[np.arange(nlines - half_alt) / nlines, 0.5]
            extract = lambda arr: arr[indices, half_act]

        fig, ax = plt.subplots(figsize=(10, 6))
        for name, label in zip(mtf_names, mtf_labels):
            y = extract(arrays[name])
            ax.plot(frequencies, y, label=label,
                    linewidth=2.5 if name == 'Hsys' else 1.2,
                    color='black' if name == 'Hsys' else None)
        ax.axvline(0.5, linestyle='--', color='gray', linewidth=1, label='Nyquist')
        ax.set(xlabel='Normalised spatial frequency [cycles/pixel]',
               ylabel='MTF', title=f'System MTF and six contributions - {band} - {direction}',
               xlim=(0, 0.5), ylim=(0, 1.05))
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, f'mtf_reference_{direction.lower()}_{band}.png'), dpi=200)
        plt.close(fig)

    # Current plotMtf stores only the last band, overwriting previous matrices.
    # Validate the saved system MTF against VNIR-3 reference (last processed band).
    if band == bands[-1]:
        own_hsys = read_toa_silent(outdir, 'Hsys.nc')
        compare_output(own_hsys, arrays['Hsys'], 'Hsys_saved_last_band.nc')
        own_hdet = read_toa_silent(outdir, 'Hdet.nc')
        compare_output(own_hdet, arrays['Hdet'], 'Hdet_saved_last_band.nc')

    for direction in ('act', 'alt'):
        own_figure = os.path.join(outdir, f'graph_mtf_{direction}_{band}.png')
        if os.path.isfile(own_figure):
            logger.info(f'{band} {direction.upper()}: own MTF figure available')
        else:
            logger.info(f'{band} {direction.upper()}: own MTF figure MISSING')
            results[f'figure_{direction}_{band}'] = 'MISSING'