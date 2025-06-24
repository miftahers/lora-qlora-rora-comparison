import numpy as np                   # Untuk operasi numerik seperti mean dan std
from scipy import stats             # Untuk fungsi statistik seperti confidence interval dan ANOVA

# --- Data hasil 5 kali pengujian untuk masing-masing metode PEFT ---
# Nilai-nilai ini bisa berupa akurasi, F1-score, atau metrik lain
lora_scores = [0.84, 0.85, 0.83, 0.84, 0.85]       # Skor LoRA dari 5 seed
qlora_scores = [0.82, 0.83, 0.82, 0.81, 0.83]      # Skor QLoRA
rora_scores = [0.80, 0.79, 0.78, 0.79, 0.80]    # Skor Rora

# --- Fungsi untuk menghitung statistik deskriptif ---
def deskriptif(scores):
    mean = np.mean(scores)  # Rata-rata dari 5 skor
    std = np.std(scores, ddof=1)  # Standar deviasi sampel (ddof=1 untuk sample, bukan populasi)
    
    # Menghitung Confidence Interval 95% menggunakan distribusi t
    ci95 = stats.t.interval(
        0.95,                      # Confidence level 95%
        df=len(scores)-1,          # Derajat kebebasan (n-1)
        loc=mean,                  # Rata-rata sebagai pusat distribusi
        scale=std / np.sqrt(len(scores))  # Error standar dari mean
    )
    
    return mean, std, ci95

# --- Menampilkan statistik untuk tiap metode ---
for nama, skor in zip(['LoRA', 'QLoRA', 'rora'], [lora_scores, qlora_scores, rora_scores]):
    mean, std, ci = deskriptif(skor)
    print(f"{nama} - Rata-rata: {mean:.4f}, Standar Deviasi: {std:.4f}, CI 95%: ({ci[0]:.4f}, {ci[1]:.4f})")

# --- Uji ANOVA satu arah ---
# Untuk mengetahui apakah perbedaan rata-rata antara ketiga metode signifikan secara statistik
f_stat, p_val = stats.f_oneway(lora_scores, qlora_scores, rora_scores)

print(f"\n[ANOVA] Statistik F = {f_stat:.4f}, p-value = {p_val:.4f}")
