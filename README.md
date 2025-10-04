# lora-qlora-rora-comparison
Code for my thesis - Comparison of Fine Tuning Technique Quality on Question Answering Task
(Miftah Firdaus, 
miftahers@upi.edu, 
2025)

# Verify GPU
Cek driver GPU mu (disini nvidia) untuk mengecek pengunaan, versi driver, dan cuda
```!nvidia-smi```

# Install Libraries
Sebelum menjalankan kode pastikan kamu sudah menginstall library python yang dibutuhkan untuk menjalankan pelatihan model ini.
```!pip install --upgrade pip```
```!pip install --upgrade transformers datasets peft bitsandbytes accelerate evaluate seqeval fsspec huggingface_hub```
