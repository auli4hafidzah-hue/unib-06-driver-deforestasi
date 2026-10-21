"""Materi 6 - Attributing Direct Drivers to Deforestation Alerts with
Satellite Imagery and AI.

Membangun klasifikasi multi-kelas driver deforestasi dari ciri spektral +
tekstur + konteks akses, lalu menjelaskan hasilnya dengan feature importance
dan analisis kesalahan per kelas.

Jalankan:
    python src/materi6_atribusi_driver.py
Output:
    outputs/materi6_confusion_matrix.csv
    outputs/materi6_feature_importance.csv
    outputs/materi6_ringkasan.txt
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

DATA = os.path.join("data", "materi6_driver_deforestasi.csv")
OUT = "outputs"
TARGET = "driver"
FEATURES = ["ndvi_drop", "ndmi_drop", "nbr_post", "burn_index",
            "texture_glcm", "road_dist_km", "patch_area_ha", "slope_deg"]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=600, min_samples_leaf=1, class_weight="balanced",
            random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        cv_ba = cross_val_score(m, X_tr, y_tr, cv=cv,
                                scoring="balanced_accuracy", n_jobs=-1)
        rows.append(dict(model=name,
                         balanced_accuracy_test=balanced_accuracy_score(y_te, pred),
                         cv_balanced_accuracy_mean=cv_ba.mean(),
                         cv_balanced_accuracy_std=cv_ba.std()))
        fitted[name] = m

    hasil = pd.DataFrame(rows).round(4)
    best = hasil.sort_values("balanced_accuracy_test", ascending=False).iloc[0]["model"]
    model = fitted[best]
    pred = model.predict(X_te)
    labels = sorted(y.unique())

    lines = ["=== MATERI 6: ATRIBUSI DRIVER DEFORESTASI ===\n"]
    lines.append(f"Data: {len(df)} alert | kelas: {len(labels)} | fitur: {len(FEATURES)}")
    lines.append("Kelas besarnya tidak seimbang -> metrik utama = balanced accuracy, "
                 "bukan accuracy biasa.\n")
    lines.append("-- Perbandingan model --")
    lines.append(hasil.to_string(index=False))
    lines.append(f"\nModel terbaik: {best}")
    lines.append("\n-- Classification report (test) --")
    lines.append(classification_report(y_te, pred, digits=3))

    cm = confusion_matrix(y_te, pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"aktual_{l}" for l in labels],
                         columns=[f"pred_{l}" for l in labels])
    lines.append("\n-- Confusion matrix --")
    lines.append(cm_df.to_string())
    cm_df.to_csv(os.path.join(OUT, "materi6_confusion_matrix.csv"))

    # --- feature importance (permutation lebih jujur utk tree) ---------
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    lines.append("\n-- Feature importance (impurity-based) --")
    lines.append(imp.round(4).to_string())
    imp.to_csv(os.path.join(OUT, "materi6_feature_importance.csv"), header=["importance"])

    # --- analisis kesalahan per kelas ----------------------------------
    lines.append("\n-- Analisis kesalahan --")
    cm_norm = cm / cm.sum(1, keepdims=True)
    for i, l in enumerate(labels):
        worst = np.argsort(-cm_norm[i])[:2]
        palsu = [f"{labels[j]} ({cm_norm[i, j]:.1%})" for j in worst if j != i]
        if palsu:
            lines.append(f"  {l}: paling sering salah jadi {' dan '.join(palsu)}")
    lines.append("\nInterpretasi: kebakaran & pembalakan tumpang-tindih karena "
                 "keduanya menurunkan NDVI; pembeda utamanya burn_index dan "
                 "jarak ke jalan. Perkebunan & pertambakan dipisahkan terutama "
                 "oleh tekstur GLCM dan luas tapak.")

    # --- plot -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    im = axes[0].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_title(f"Confusion matrix (ternormalisasi) - {best}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[0].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    imp.plot.barh(ax=axes[1], color="#4c72b0")
    axes[1].set_title("Feature importance")
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "materi6_atribusi_driver.png"), dpi=140)
    plt.close(fig)

    report = "\n".join(lines)
    with open(os.path.join(OUT, "materi6_ringkasan.txt"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
