from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.ismConfig import ismConfig


def main():
    cfg = ismConfig()
    h = 6.62607015e-34  # Planck constant [J s]
    c = 299792458.0     # Speed of light [m/s]

    k_rad_irrad = cfg.Tr * np.pi / 4 * (cfg.D / cfg.f) ** 2
    area_pix = cfg.pix_size ** 2
    k_irrad_phot = 1e-3 * area_pix * cfg.t_int * cfg.wv / (h * c)
    k_phot_electr = cfg.QE
    k_electr_volt = cfg.OCF * cfg.ADC_gain
    k_volt_dn = (2 ** cfg.bit_depth - 1) / (cfg.max_voltage - cfg.min_voltage)
    k_total = k_rad_irrad * k_irrad_phot * k_phot_electr * k_electr_volt * k_volt_dn

    print('-' * 92)
    print('ISM - ANALYTICAL CONVERSION FACTORS (NOMINAL LINEAR CHAIN)')
    print('-' * 92)
    print(f"{'Conversion':<30}" + ''.join(f'{"VNIR-" + str(i):>15}' for i in range(4)))
    rows = [
        ('Radiance -> Irradiance', np.full(4, k_rad_irrad)),
        ('Irradiance -> Photons', k_irrad_phot),
        ('Photons -> Electrons', np.full(4, k_phot_electr)),
        ('Electrons -> Volts', np.full(4, k_electr_volt)),
        ('Volts -> DN', np.full(4, k_volt_dn)),
        ('TOTAL: Radiance -> DN', k_total),
    ]
    for label, values in rows:
        print(f'{label:<30}' + ''.join(f'{v:>15.6g}' for v in values))
    print('-' * 92)
    print('Note: the total factor excludes spectral integration, spatial filtering,')
    print('PRNU, dark signal, bad/dead pixels, FWC clipping and DN quantisation.')
    print('Units: radiance [mW/(m^2 sr)] -> irradiance [mW/m^2] -> photons ->')
    print('electrons -> volts -> digital counts. The radiance-to-irradiance')
    print('coefficient carries the corresponding solid-angle conversion.')


if __name__ == '__main__':
    main()
