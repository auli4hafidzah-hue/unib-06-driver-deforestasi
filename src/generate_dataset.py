"""Generator dataset untuk repo ini.

Data sintetis dengan seed tetap (reproducible), tidak memuat data pribadi.
Jalankan:  python src/generate_dataset.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from common import seed_default

SEED = seed_default()


def gen_deforestation(n: int = 4000, seed: int = SEED) -> pd.DataFrame:
    """Klasifikasi driver deforestasi dari ciri spektral + konteks.

    Fitur sengaja dibuat punya pemisahan kelas yang realistis namun tidak
    sepenuhnya linear: ada tumpang-tindih antara kebakaran dan pembalakan
    karena keduanya menurunkan NDVI.
    """
    rng = np.random.default_rng(seed)
    drivers = np.array(["pembalakan", "perkebunan", "kebakaran", "pertambangan"])
    prior = np.array([0.32, 0.34, 0.21, 0.13])
    y = rng.choice(len(drivers), size=n, p=prior)

    rows = []
    for i, cls in enumerate(y):
        d = drivers[cls]
        if d == "pembalakan":
            ndvi_drop = rng.normal(0.55, 0.10)
            ndmi_drop = rng.normal(0.30, 0.09)
            texture = rng.normal(0.72, 0.11)          # tapal batas tak beraturan
            road_dist = abs(rng.normal(1.8, 1.1))     # dekat jalan = akses
            burn_index = rng.normal(0.12, 0.06)
            patch_area = abs(rng.normal(14, 8))       # tapak kecil-menengah
            slope = abs(rng.normal(12, 7))
        elif d == "perkebunan":
            ndvi_drop = rng.normal(0.42, 0.09)
            ndmi_drop = rng.normal(0.18, 0.08)
            texture = rng.normal(0.28, 0.10)          # blok rapi, geometris
            road_dist = abs(rng.normal(5.5, 2.5))
            burn_index = rng.normal(0.05, 0.04)
            patch_area = abs(rng.normal(48, 22))      # blok besar
            slope = abs(rng.normal(8, 5))
        elif d == "kebakaran":
            ndvi_drop = rng.normal(0.60, 0.12)
            ndmi_drop = rng.normal(0.34, 0.10)
            texture = rng.normal(0.45, 0.12)
            road_dist = abs(rng.normal(9.0, 4.0))     # biasanya jauh dari jalan
            burn_index = rng.normal(0.78, 0.11)       # NBR sangat rendah
            patch_area = abs(rng.normal(70, 30))
            slope = abs(rng.normal(6, 4))
        else:  # pertambangan
            ndvi_drop = rng.normal(0.70, 0.11)
            ndmi_drop = rng.normal(0.40, 0.10)
            texture = rng.normal(0.88, 0.08)          # sangat tak beraturan
            road_dist = abs(rng.normal(14.0, 6.0))
            burn_index = rng.normal(0.08, 0.05)
            patch_area = abs(rng.normal(95, 40))      # tapak sangat besar
            slope = abs(rng.normal(18, 8))

        rows.append(
            dict(
                alert_id=f"A{i:05d}",
                driver=d,
                ndvi_drop=float(np.clip(ndvi_drop, 0, 1)),
                ndmi_drop=float(np.clip(ndmi_drop, 0, 1)),
                nbr_post=float(np.clip(0.55 - burn_index * 0.6 + rng.normal(0, 0.03), -1, 1)),
                burn_index=float(np.clip(burn_index, 0, 1)),
                texture_glcm=float(np.clip(texture, 0, 1)),
                road_dist_km=float(road_dist),
                patch_area_ha=float(patch_area),
                slope_deg=float(slope),
                region=rng.choice(["Sumatera", "Kalimantan", "Papua", "Sulawesi"]),
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- materi 1

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=seed_default())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = gen_deforestation(seed=a.seed)
    path = os.path.join(a.out, "materi6_driver_deforestasi.csv")
    df.to_csv(path, index=False)
    print(f"{path}  {len(df)} baris, {len(df.columns)} kolom")


if __name__ == "__main__":
    main()
