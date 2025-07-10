# 🖼️ Reversible Image Watermarking using DWT-SVD and Arnold Transform

This project implements a **robust and reversible image watermarking** technique using:
- **3-level Discrete Wavelet Transform (DWT)**
- **Singular Value Decomposition (SVD)**
- **Arnold Transform** for watermark encryption
- **Alpha optimization** for balancing imperceptibility and robustness
- Comprehensive **robustness testing** under various image attacks


## 📌 Features

- ✅ Multi-level DWT decomposition (LL3 band used for embedding)
- ✅ SVD-based watermark embedding with automatic alpha tuning
- ✅ Arnold scrambling for watermark security
- ✅ Reversible host recovery after extraction
- ✅ Robustness simulation under various attacks
- ✅ Quality metrics: PSNR, SSIM, and NC


## 🧰 Requirements

Install all required dependencies using pip:

```pip install numpy opencv-python pywavelets scikit-image```
