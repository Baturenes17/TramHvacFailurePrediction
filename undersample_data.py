# -*- coding: utf-8 -*-
"""
Random Undersampling — 3_years_data_no7d.csv referans alinarak.

Cogunluk sinifini (failure_next_30d=0) rastgele azaltarak azinlik sinifina
(failure_next_30d=1) esitler (1:1 tam denge). Satir bazli calisir; orijinal
metin formati (';' ayrac, ondalik virgul, tarih formati) birebir korunur.

Cikti: 3_years_data_no7d_undersampled.csv
"""
import csv
import random

SRC = "3_years_data_no7d.csv"
DST = "3_years_data_no7d_undersampled.csv"
TARGET_COL = "failure_next_30d"
RATIO = 1  # cogunluk : azinlik orani (1 => 1:1 tam denge)
SEED = 42

random.seed(SEED)

# Ham satirlari oku (formati bozmamak icin yeniden yazmiyoruz)
with open(SRC, "r", encoding="utf-8-sig", newline="") as f:
    lines = f.read().splitlines()

header = lines[0]
cols = header.split(";")
tgt_idx = cols.index(TARGET_COL)

# Indeksleri sinifa gore ayir (duplike satirlarda sorun olmamasi icin)
data = [line for line in lines[1:] if line.strip()]
pos_idx, neg_idx = [], []
for i, line in enumerate(data):
    val = line.split(";")[tgt_idx].strip()
    (pos_idx if val == "1" else neg_idx).append(i)

n_pos = len(pos_idx)
keep_neg = min(len(neg_idx), RATIO * n_pos)
neg_sampled = set(random.sample(neg_idx, keep_neg))

# Orijinal sirayi koru (tarih/arac sirasi bozulmasin)
kept = set(pos_idx) | neg_sampled
out_lines = [data[i] for i in range(len(data)) if i in kept]
neg = neg_idx

with open(DST, "w", encoding="utf-8-sig", newline="") as f:
    f.write(header + "\n")
    f.write("\n".join(out_lines) + "\n")

print(f"Kaynak     : {SRC}")
print(f"  0 (cogunluk): {len(neg)}")
print(f"  1 (azinlik) : {n_pos}")
print(f"Cikti      : {DST}")
print(f"  0 (cogunluk): {keep_neg}")
print(f"  1 (azinlik) : {n_pos}")
print(f"  toplam      : {keep_neg + n_pos}")
