# Studi Kasus 6 — Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI

Materi dari [Informatika UNIB Official](https://youtube.com/@informatikaunibofficial):
**"Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI"**.

Studi kasus klasifikasi multi-kelas driver deforestasi dari ciri spektral, tekstur,
dan konteks akses — mengubah alert kebakaran/penebangan menjadi atribusi sebab.

> Data **sintetis dengan seed tetap** (reproducible). Fitur dibuat meniru ciri yang
> dipakai di literatur (NDVI, NBR, GLCM, jarak jalan), tapi nilainya bukan hasil
> pengolahan citra satelit sungguhan.

## Menjalankan

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash run.sh          # atau:
python src/generate_dataset.py
python src/materi6_atribusi_driver.py
```

Hasil masuk ke `outputs/`.

## Dataset

`data/materi6_driver_deforestasi.csv` — 4.000 alert, 4 kelas.

| Kelas | Proporsi |
|---|---|
| `perkebunan` | 34% |
| `pembalakan` | 32% |
| `kebakaran` | 21% |
| `pertambangan` | 13% |

Fitur (8): `ndvi_drop`, `ndmi_drop`, `nbr_post`, `burn_index`, `texture_glcm`,
`road_dist_km`, `patch_area_ha`, `slope_deg`.

Tumpang-tindih antarkelas **sengaja dibuat**: kebakaran dan pembalakan sama-sama
menurunkan NDVI, sehingga keduanya hanya bisa dipisahkan oleh `burn_index` dan
jarak ke jalan — bukan oleh kerusaman kanopi saja.

## Metode

1. **Dua model** — Random Forest (600 pohon, `class_weight="balanced"`) dan
   Gradient Boosting.
2. **Metrik utama = balanced accuracy**, bukan accuracy biasa, karena kelas
   tidak seimbang (13% vs 34%).
3. **Validasi** — stratified 5-fold CV, evaluasi pada test set 25%.
4. **Interpretasi** — confusion matrix ternormalisasi, feature importance, dan
   analisis kesalahan per kelas.

## Hasil terakhir

| Model | Balanced accuracy test | CV mean | CV std |
|---|---|---|---|
| random_forest | **0,9976** | — | — |
| gradient_boosting | — | — | — |

Confusion matrix (Random Forest):

| aktual \ pred | kebakaran | pembalakan | perkebunan | pertambangan |
|---|---|---|---|---|
| kebakaran | **214** | 0 | 0 | 0 |
| pembalakan | 0 | **322** | 2 | 0 |
| perkebunan | 0 | 1 | **335** | 0 |
| pertambakan | 0 | 0 | 0 | **126** |

Feature importance:

| Fitur | Importance |
|---|---|
| `texture_glcm` | 0,237 |
| `nbr_post` | 0,170 |
| `burn_index` | 0,159 |
| `road_dist_km` | 0,158 |
| `patch_area_ha` | 0,129 |
| `ndvi_drop` | 0,068 |
| `ndmi_drop` | 0,058 |
| `slope_deg` | 0,021 |

## Temuan penting

- **Kebakaran terklasifikasi sempurna** (214/214) karena `burn_index` adalah
  pembeda yang sangat kuat — memang ciri khas kebakaran.
- Kebingungan yang tersisa hanya **pembalakan ↔ perkebunan** (2 dan 1 kasus).
  Ini sesuai dugaan: keduanya menurunkan NDVI dan terjadi di wilayah berakses jalan.
- Fitur terpenting adalah **tekstur GLCM** — konsisten dengan literatur bahwa
  aktivitas buatan manusia (perkebunan, pertambakan) punya pola keruangan yang
  lebih teratur/geometris dibanding kerusakan alami.
- **`ndvi_drop` sendiri hanya menyumbang 0,068** pentingnya. Artinya atribusi
  driver tidak bisa bergantung pada indeks vegetasi tunggal; konteks keruangan
  (tekstur, luas tapak, akses) justru yang memisahkan kelas.

Ini inti materi: **atribusi sebab butuh lebih dari sekadar mendeteksi "ada yang hilang".**

## Struktur

```
src/common.py                    # helper path & direktori
src/generate_dataset.py          # generator alert sintetis
src/materi6_atribusi_driver.py
data/    outputs/                # dibuat saat runtime, tidak di-commit
```

## Referensi konsep

- Haralick, R. M., et al. (1973). Textural Features for Image Classification (GLCM).
- Brodersen, K. H., et al. (2010). The Balanced Accuracy and Its Posterior Distribution.

## Lisensi

MIT — lihat [LICENSE](LICENSE).
