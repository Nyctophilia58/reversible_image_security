import numpy as np
import cv2
import pywt
import os
import time
from skimage.metrics import structural_similarity as ssim


def resize_and_gray(path, size):
    """Load and resize an image to grayscale."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.resize(img, size)

def normalize_image(img):
    """Normalize image to [0, 1]."""
    return img.astype(np.float64) / 255.0

def denormalize_image(img):
    """Denormalize image to [0, 255] and convert to uint8."""
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def apply_dwt_levels(img, level=3, wavelet='haar'):
    """Apply DWT to the image."""
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    return coeffs

def reconstruct_dwt(coeffs, wavelet='haar'):
    """Reconstruct image from DWT coefficients."""
    return pywt.waverec2(coeffs, wavelet=wavelet)

def apply_svd(img):
    """Apply SVD to an image or band."""
    U, S, Vt = np.linalg.svd(img, full_matrices=False)
    return U, S, Vt

def embed_svd(U, S, Vt, wm_U, wm_S, wm_Vt, alpha):
    """Embed watermark using SVD."""
    S_embed = S + alpha * wm_S
    return U @ np.diag(S_embed) @ Vt

def extract_svd(orig_S, watermarked_S, wm_U, wm_Vt, alpha):
    """Extract watermark using SVD."""
    if alpha == 0:
        return np.zeros_like(wm_U @ np.diag(wm_Vt))
    wm_S_recovered = (watermarked_S - orig_S) / alpha
    return wm_U @ np.diag(wm_S_recovered) @ wm_Vt

def calculate_nc(w1, w2):
    """Calculate Normalized Correlation (NC) between two images."""
    w1 = w1.astype(np.float64).flatten()
    w2 = w2.astype(np.float64).flatten()
    num = np.sum(w1 * w2)
    den = np.sqrt(np.sum(w1 ** 2) * np.sum(w2 ** 2))
    return num / den if den != 0 else 0

def safe_psnr(img1, img2, data_range=255):
    """Calculate PSNR, handling zero MSE case."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return np.inf
    return 10 * np.log10((data_range ** 2) / mse)

def arnold_scramble(image, n_iterations, n):
    """Scramble the image using Arnold Transform."""
    height, width = image.shape
    if height != n or width != n:
        raise ValueError(f"Image size ({height}x{width}) must match N ({n})")
    scrambled = np.zeros_like(image, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            x = i
            y = j
            for _ in range(n_iterations):
                x_new = (x + y) % n
                y_new = (x + 2 * y) % n
                x = x_new
                y = y_new
            scrambled[x, y] = image[i, j]
    return scrambled

def arnold_unscramble(image, n_iterations, n):
    """Unscramble the image using the reverse Arnold Transform."""
    height, width = image.shape
    if height != n or width != n:
        raise ValueError(f"Image size ({height}x{width}) must match N ({n})")
    unscrambled = np.zeros_like(image, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            x = i
            y = j
            for _ in range(n_iterations):
                x_new = (2 * x - y) % n
                y_new = (-x + y) % n
                x = x_new
                y = y_new
            unscrambled[x, y] = image[i, j]
    return unscrambled

def embed_watermark_band(host_band, wm_img, alpha):
    """Embed watermark into a DWT band using SVD."""
    wm_U, wm_S, wm_Vt = apply_svd(wm_img)
    host_U, host_S, host_Vt = apply_svd(host_band)
    wm_band = embed_svd(host_U, host_S, host_Vt, wm_U, wm_S, wm_Vt, alpha)
    return wm_band, host_S, wm_U, wm_S, wm_Vt, host_U, host_Vt

def extract_watermark_band(watermarked_band, orig_S, wm_U, wm_Vt, alpha):
    """Extract watermark from a watermarked DWT band."""
    _, S_wm, _ = np.linalg.svd(watermarked_band, full_matrices=False)
    wm_extracted = extract_svd(orig_S, S_wm, wm_U, wm_Vt, alpha)
    return wm_extracted

def normalize_metric(value, min_val, max_val):
    """Normalize a metric to [0, 1] range."""
    return (value - min_val) / (max_val - min_val) if max_val != min_val else 1.0

def apply_gaussian_noise(image, noise_std, seed=None):
    """Apply Gaussian noise to an image and return the result."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a uint8 NumPy array.")
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(0, noise_std, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def apply_poisson_noise(image, scale_factor, seed=None):
    """Apply Poisson noise to an image and return the result."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a uint8 NumPy array.")
    if seed is not None:
        np.random.seed(seed)
    img_float = image.astype(np.float64)
    img_scaled = img_float * scale_factor
    noisy_image = np.random.poisson(img_scaled)
    noisy_image = np.clip(noisy_image / scale_factor, 0, 255).astype(np.uint8)
    return noisy_image

def apply_salt_and_pepper_noise(image, prob, seed=None):
    """Apply salt-and-pepper noise to an image and return the result."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a uint8 NumPy array.")
    if seed is not None:
        np.random.seed(seed)
    output = np.copy(image)
    noise = np.random.random(image.shape)
    output[noise < prob / 2] = 255
    output[(noise >= prob / 2) & (noise < prob)] = 0
    return output

def apply_speckle_noise(image, variance, seed=None):
    """Apply speckle noise to an image and return the result."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a uint8 NumPy array.")
    if seed is not None:
        np.random.seed(seed)
    img_float = image.astype(np.float64) / 255.0
    noise = np.random.normal(0, np.sqrt(variance), image.shape)
    noisy_image = img_float + img_float * noise
    noisy_image = np.clip(noisy_image * 255.0, 0, 255).astype(np.uint8)
    return noisy_image

def apply_cropping(image, crop_percentage, mode='replicate', seed=None):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input must be a uint8 NumPy array.")
    if not (0 <= crop_percentage <= 0.5):
        raise ValueError("crop_percentage must be in [0, 0.5]")
    h, w = image.shape
    crop_h = int(h * crop_percentage)
    crop_w = int(w * crop_percentage)
    if seed is not None:
        np.random.seed(seed)
    start_y = np.random.randint(0, crop_h + 1)
    start_x = np.random.randint(0, crop_w + 1)
    end_y = h - (crop_h - start_y)
    end_x = w - (crop_w - start_x)
    cropped = image[start_y:end_y, start_x:end_x]
    pad_top = start_y
    pad_bottom = h - end_y
    pad_left = start_x
    pad_right = w - end_x
    if mode == 'zero':
        padded = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=0)
    elif mode == 'replicate':
        padded = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REPLICATE)
    else:
        raise ValueError("Invalid mode. Use 'zero' or 'replicate'.")
    return padded

def apply_rotation(image, angle):
    """Apply rotation to an image and return the result."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a uint8 NumPy array.")
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), borderValue=0)
    return rotated_image

def apply_scaling(image, scale_factor):
    """Apply scaling attack by resizing down and back up to original size."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a uint8 NumPy array.")
    if not 0 < scale_factor < 1:
        raise ValueError("Scale factor must be between 0 and 1")
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    scaled_down = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scaled_image = cv2.resize(scaled_down, (w, h), interpolation=cv2.INTER_CUBIC)
    return scaled_image

def apply_jpeg_compression(image, quality):
    """Apply JPEG compression to an image and return the result."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a uint8 NumPy array.")
    temp_path = "temp_jpeg_compressed.jpg"
    cv2.imwrite(temp_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    compressed_image = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    os.remove(temp_path)
    if compressed_image is None:
        raise ValueError("Failed to load compressed image.")
    return compressed_image

def apply_gaussian_blur(image, sigma, kernel_size=5):
    """Apply Gaussian blur attack to an image."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a uint8 NumPy array.")
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Kernel size must be an odd integer >= 3")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)

def apply_median_blur(image, kernel_size=5):
    """Apply median blur attack to an image."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a uint8 NumPy array.")
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Kernel size must be an odd integer >= 3")
    return cv2.medianBlur(image, kernel_size)

def apply_average_blur(image, kernel_size=5):
    """Apply average blur (mean filter) attack to an image."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a uint8 NumPy array.")
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Kernel size must be an odd integer >= 3")
    return cv2.blur(image, (kernel_size, kernel_size))

def apply_hybrid_attack_1(image, noise_std=0.1, jpeg_quality=70, seed=None):
    """Apply hybrid attack: Gaussian noise followed by JPEG compression."""
    noisy_image = apply_gaussian_noise(image, noise_std, seed)
    compressed_image = apply_jpeg_compression(noisy_image, jpeg_quality)
    return compressed_image

def apply_hybrid_attack_2(image, prob=0.001, angle=0.01, seed=None):
    """Apply hybrid attack: Salt-and-pepper noise followed by rotation."""
    noisy_image = apply_salt_and_pepper_noise(image, prob, seed)
    rotated_image = apply_rotation(noisy_image, angle)
    return rotated_image

def apply_hybrid_attack_3(image, prob=0.001, sigma=0.5, kernel_size=3, seed=None):
    """Apply hybrid attack: Salt-and-pepper noise followed by Gaussian blur."""
    noisy_image = apply_salt_and_pepper_noise(image, prob, seed)
    blurred_image = apply_gaussian_blur(noisy_image, sigma, kernel_size)
    return blurred_image

def apply_hybrid_attack_4(image, crop_percentage=0.03, quality=70, seed=None):
    """Apply hybrid attack: Cropping followed by JPEG compression."""
    cropped_image = apply_cropping(image, crop_percentage, mode='replicate', seed=seed)
    compressed_image = apply_jpeg_compression(cropped_image, quality)
    return compressed_image

def apply_hybrid_attack_5(image, variance=0.001, prob=0.001, seed=None):
    """Apply hybrid attack: Speckle noise followed by salt-and-pepper noise."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a uint8 NumPy array.")
    if seed is not None:
        np.random.seed(seed)
    speckled_image = apply_speckle_noise(image, variance, seed)
    hybrid_image = apply_salt_and_pepper_noise(speckled_image, prob, seed)
    return hybrid_image

def recover_host_band(watermarked_band, wm_U, wm_S, wm_Vt, alpha):
    """Recover the original host band from the watermarked band using inverse SVD embedding."""
    U_wm, S_wm, Vt_wm = np.linalg.svd(watermarked_band, full_matrices=False)
    S_recovered = S_wm - alpha * wm_S
    return U_wm @ np.diag(S_recovered) @ Vt_wm

def objective_function(alpha, host_band, wm_norm, host, coeffs_copy):
    try:
        # Create a fresh copy of coeffs
        coeffs_temp = [np.copy(c) if isinstance(c, np.ndarray) else tuple(np.copy(sub) for sub in c) for c in coeffs_copy]

        # Embed watermark
        wm_band, orig_S, wm_U, wm_S, wm_Vt, host_U, host_Vt = embed_watermark_band(host_band, wm_norm, alpha)
        coeffs_temp[0] = wm_band
        watermarked = reconstruct_dwt(coeffs_temp)
        watermarked_uint8 = denormalize_image(watermarked)

        # Extract watermark
        recoeffs = apply_dwt_levels(normalize_image(watermarked_uint8), level=3)
        re_band = recoeffs[0]
        wm_extracted = extract_watermark_band(re_band, orig_S, wm_U, wm_Vt, alpha)

        # Calculate metrics
        psnr_wm = safe_psnr(host, watermarked_uint8, data_range=255)
        ssim_wm = ssim(host, watermarked_uint8, data_range=255)
        nc = calculate_nc(wm_norm, denormalize_image(wm_extracted))

        # Normalize metrics
        psnr_norm = normalize_metric(psnr_wm, 20, 70)
        ssim_norm = ssim_wm
        nc_norm = nc

        # Apply penalties for failing thresholds
        penalty = 0
        if psnr_wm < 35:
            penalty += (35 - psnr_wm) * 0.1
        if ssim_wm < 0.85:
            penalty += (0.85 - ssim_wm) * 10
        if nc < 0.85:
            penalty += (0.85 - nc) * 10

        # Weighted objective (maximize)
        score = 0.3 * psnr_norm + 0.2 * ssim_norm + 0.5 * nc_norm - penalty

        return score, psnr_wm, ssim_wm, nc
    except Exception as e:
        print(f"Error for alpha={alpha}: {e}")
        return -np.inf, 0, 0, 0

def optimize_alpha(host_band, wm_norm, host, coeffs_copy, alpha_range=(0.01, 2.0), step=0.01):
    alpha_values = np.arange(alpha_range[0], alpha_range[1] + step, step)
    best_alpha = alpha_values[0]
    best_score = -np.inf
    results = []
    best_metrics = (0, 0, 0)  # (PSNR, SSIM, NC)

    for alpha in alpha_values:
        score, psnr_wm, ssim_wm, nc = objective_function(alpha, host_band, wm_norm, host, coeffs_copy)
        meets_thresholds = psnr_wm >= 35 and ssim_wm >= 0.85 and nc >= 0.85
        results.append((alpha, score, psnr_wm, ssim_wm, nc, meets_thresholds))
        if score > best_score and not np.isnan(score):
            best_score = score
            best_alpha = alpha
            best_metrics = (psnr_wm, ssim_wm, nc)

    valid_alphas = [(a, s, (p, ss, n)) for a, s, p, ss, n, mt in results if mt]
    if valid_alphas:
        valid_psnrs, valid_ssims, valid_ncs, valid_alphas_only = zip(
            *[(m[0], m[1], m[2], a) for a, _, m in valid_alphas])
        median_psnr, median_ssim, median_nc = np.median(valid_psnrs), np.median(valid_ssims), np.median(valid_ncs)
        min_psnr, max_psnr = min(valid_psnrs), max(valid_psnrs)
        min_ssim, max_ssim = min(valid_ssims), max(valid_ssims)
        min_nc, max_nc = min(valid_ncs), max(valid_ncs)

        min_distance = np.inf
        balanced_alpha = valid_alphas_only[0]
        balanced_metrics = valid_alphas[0][2]

        for alpha, _, metrics in valid_alphas:
            psnr_norm = normalize_metric(metrics[0], min_psnr, max_psnr)
            ssim_norm = normalize_metric(metrics[1], min_ssim, max_ssim)
            nc_norm = normalize_metric(metrics[2], min_nc, max_nc)
            median_psnr_norm = normalize_metric(median_psnr, min_psnr, max_psnr)
            median_ssim_norm = normalize_metric(median_ssim, min_ssim, max_ssim)
            median_nc_norm = normalize_metric(median_nc, min_nc, max_nc)

            distance = np.sqrt(
                (psnr_norm - median_psnr_norm) ** 2 +
                (ssim_norm - median_ssim_norm) ** 2 +
                (nc_norm - median_nc_norm) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                balanced_alpha = alpha
                balanced_metrics = metrics

        return balanced_alpha, valid_alphas[0][1], balanced_metrics, results
    else:
        print("\nNo valid alphas meet all thresholds. Returning best alpha by score.")
        return best_alpha, best_score, best_metrics, results

# === MAIN EXECUTION ===
host_path = "images/area.tiff"
# host_path = "images/brain.jpeg"
# host_path = "images/ct-scan.jpg"
# host_path = "images/girl.jpg"
# host_path = "images/grass.tiff"
# host_path = "images/house.tiff"
# host_path = "images/pepper.tiff"
# host_path = "images/texture.tiff"
# host_path = "images/tribe_woman.jpg"

wm_path = "images/watermark.jpg"
output_folder = "output_folder"
os.makedirs(output_folder, exist_ok=True)

# Arnold Transform parameters
N = 64  # Size of the watermark (must match wm.shape)

try:
    # Load and preprocess images
    host = resize_and_gray(host_path, (512, 512))
    wm = resize_and_gray(wm_path, (64, 64))
    host_norm = normalize_image(host)
    wm_norm = normalize_image(wm)

    # Validate image sizes
    if host.shape != (512, 512) or wm.shape != (64, 64):
        raise ValueError(f"Unexpected image sizes: host={host.shape}, watermark={wm.shape}")

    # Optimize k for Arnold Transform
    optimal_k = 10

    # Apply Arnold scrambling to the blurred watermark
    wm_norm_scrambled = arnold_scramble(wm_norm * 255, optimal_k, N) / 255.0

    # Apply DWT and process LL3 band
    coeffs = apply_dwt_levels(host_norm, level=3)
    coeffs_copy = [np.copy(c) if isinstance(c, np.ndarray) else tuple(np.copy(sub) for sub in c) for c in coeffs]
    host_band = coeffs_copy[0]

    # Optimize alpha using grid search
    optimal_alpha, best_score, best_metrics, results = optimize_alpha(host_band, wm_norm_scrambled, host, coeffs_copy)

    # Check if thresholds are met
    if best_metrics[0] >= 35 and best_metrics[1] >= 0.85 and best_metrics[2] >= 0.85:
        print("All thresholds met (PSNR >= 35, SSIM >= 0.85, NC >= 0.85)")
    else:
        print("Warning: No alpha meets all thresholds. Selected best possible alpha.")

    # Embed watermark with optimal alpha
    wm_band, orig_S, wm_U, wm_S, wm_Vt, host_U, host_Vt = embed_watermark_band(host_band, wm_norm_scrambled, optimal_alpha)
    coeffs_copy[0] = wm_band

    # Reconstruct watermarked image
    watermarked = reconstruct_dwt(coeffs_copy)
    watermarked_uint8 = denormalize_image(watermarked)

    psnr_wm = safe_psnr(host, watermarked_uint8, data_range=255)
    ssim_wm = ssim(host, watermarked_uint8, data_range=255)

    """
    # Apply Gaussian noise
    watermarked_uint8 = apply_gaussian_noise(watermarked_uint8, noise_std=7, seed=42)
    noisy_output_path = os.path.join(output_folder, "LL3_watermarked_gaussian.png")
    cv2.imwrite(noisy_output_path, watermarked_uint8)
    """

    """
    # --- Apply Poisson noise ---
    watermarked_uint8 = apply_poisson_noise(watermarked_uint8, scale_factor=50.0, seed=42)
    poisson_output_path = os.path.join(output_folder, "LL3_watermarked_poisson.jpg")
    cv2.imwrite(poisson_output_path, watermarked_uint8)
    """

    """
    # --- Apply Salt and Pepper attack ---
    watermarked_uint8 = apply_salt_and_pepper_noise(watermarked_uint8, prob=0.002, seed=42)
    noisy_output_path = os.path.join(output_folder, "LL3_watermarked_salt_pepper.jpg")
    cv2.imwrite(noisy_output_path, watermarked_uint8)
    """

    """
    # --- Apply speckle noise ---
    watermarked_uint8 = apply_speckle_noise(watermarked_uint8, variance=0.005, seed=42)
    noisy_output_path = os.path.join(output_folder, "LL3_watermarked_speckle.jpg")
    cv2.imwrite(noisy_output_path, watermarked_uint8)
    """

    """
    # --- Apply cropping attack ---
    watermarked_uint8 = apply_cropping(watermarked_uint8, crop_percentage=0.005, seed=42)
    cropped_output_path = os.path.join(output_folder, "LL3_watermarked_cropped.jpg")
    cv2.imwrite(cropped_output_path, watermarked_uint8)
    """

    """
    # --- Apply rotation attack ---
    watermarked_uint8 = apply_rotation(watermarked_uint8, angle=0.01)
    rotated_output_path = os.path.join(output_folder, "LL3_watermarked_rotated.jpg")
    cv2.imwrite(rotated_output_path, watermarked_uint8)
    """

    """
    # --- Apply scaling attack ---
    watermarked_uint8 = apply_scaling(watermarked_uint8, scale_factor=0.9)
    scaled_output_path = os.path.join(output_folder, "LL3_watermarked_scaled.jpg")
    cv2.imwrite(scaled_output_path, watermarked_uint8)
    """

    """
    # --- Apply JPEG compression ---
    watermarked_uint8 = apply_jpeg_compression(watermarked_uint8, quality=40)
    compressed_output_path = os.path.join(output_folder, "LL3_watermarked_jpeg.jpg")
    cv2.imwrite(compressed_output_path, watermarked_uint8)
    """

    """
    # --- Apply Filtering attack (gaussian blur) ---
    watermarked_uint8 = apply_gaussian_blur(watermarked_uint8, sigma=1, kernel_size=5)
    blurred_output_path = os.path.join(output_folder, "LL3_watermarked_gaussian_blurred.jpg")
    cv2.imwrite(blurred_output_path, watermarked_uint8)
    """

    """
    # --- Apply Filtering attack (median) ---
    watermarked_uint8 = apply_median_blur(watermarked_uint8, kernel_size=7)
    blurred_output_path = os.path.join(output_folder, "LL3_watermarked_median_blurred.jpg")
    cv2.imwrite(blurred_output_path, watermarked_uint8)
    """

    """
    # --- Apply Filtering attack (average blur) ---
    watermarked_uint8 = apply_average_blur(watermarked_uint8, kernel_size=7)
    blurred_output_path = os.path.join(output_folder, "LL3_watermarked_average_blurred.jpg")
    cv2.imwrite(blurred_output_path, watermarked_uint8)
    """

    """
    # --- Apply Hybrid Attack 1 (Gaussian noise + JPEG compression) ---
    watermarked_uint8 = apply_hybrid_attack_1(watermarked_uint8, noise_std=7, jpeg_quality=50, seed=42)
    hybrid1_output_path = os.path.join(output_folder, "LL3_watermarked_hybrid1_gaussian_jpeg.jpg")
    cv2.imwrite(hybrid1_output_path, watermarked_uint8)
    """

    """
    # --- Apply Hybrid Attack 2 (Salt-and-pepper noise + Rotation) ---
    watermarked_uint8 = apply_hybrid_attack_2(watermarked_uint8, prob=0.003, angle=0.5, seed=42)
    hybrid2_output_path = os.path.join(output_folder, "LL3_watermarked_hybrid2_salt_pepper_rotation.jpg")
    cv2.imwrite(hybrid2_output_path, watermarked_uint8)
    """

    """
    # --- Apply Hybrid Attack 3 (Salt-and-pepper noise + Gaussian blur) ---
    watermarked_uint8 = apply_hybrid_attack_3(watermarked_uint8, prob=0.003, sigma=1, kernel_size=5, seed=42)
    hybrid3_output_path = os.path.join(output_folder, "LL3_watermarked_hybrid3_salt_pepper_gaussian.jpg")
    cv2.imwrite(hybrid3_output_path, watermarked_uint8)
    """

    """
    # --- Apply Hybrid Attack 4 (Cropping + JPEG compression) ---
    watermarked_uint8 = apply_hybrid_attack_4(watermarked_uint8, crop_percentage=0.03, quality=80, seed=42)
    hybrid4_output_path = os.path.join(output_folder, "LL3_watermarked_hybrid4_cropping_jpeg.jpg")
    cv2.imwrite(hybrid4_output_path, watermarked_uint8)
    """

    """
    # --- Apply Hybrid Attack 5 (Speckle noise + Salt-and-pepper noise) ---
    watermarked_uint8 = apply_hybrid_attack_5(watermarked_uint8, variance=0.005, prob=0.005, seed=42)
    hybrid5_output_path = os.path.join(output_folder, "LL3_watermarked_hybrid5_speckle_salt_pepper.jpg")
    cv2.imwrite(hybrid5_output_path, watermarked_uint8)
    """

    # Extract watermark from watermarked image
    recoeffs = apply_dwt_levels(normalize_image(watermarked_uint8), level=3)
    re_band = recoeffs[0]
    wm_extracted = extract_watermark_band(re_band, orig_S, wm_U, wm_Vt, optimal_alpha)

    # Apply Arnold unscrambling to the extracted watermark
    wm_extracted_unscrambled = arnold_unscramble(wm_extracted * 255, optimal_k, N) / 255.0

    # Recover the original LL3 band from the noisy, watermarked image
    recovered_LL3 = recover_host_band(re_band, wm_U, wm_S, wm_Vt, optimal_alpha)

    # Replace LL3 with recovered band
    coeffs_copy_recovered = [np.copy(c) if isinstance(c, np.ndarray) else tuple(np.copy(sub) for sub in c) for c in recoeffs]
    coeffs_copy_recovered[0] = recovered_LL3
    recovered_host = reconstruct_dwt(coeffs_copy_recovered)
    recovered_host_uint8 = denormalize_image(recovered_host)

    # Calculate metrics
    psnr_rec = safe_psnr(host, recovered_host_uint8, data_range=255)
    ssim_rec = ssim(host, recovered_host_uint8, data_range=255)
    nc = calculate_nc(wm_norm, wm_extracted_unscrambled)
    # Save results
    cv2.imwrite(os.path.join(output_folder, "watermarked.png"), watermarked_uint8)
    cv2.imwrite(os.path.join(output_folder, "reconstructed.png"), recovered_host_uint8)
    cv2.imwrite(os.path.join(output_folder, "extracted_watermark.png"), denormalize_image(wm_extracted_unscrambled))

    end_time = time.time()
    print(f"\n--- Execution Results ---")
    print(f"Optimal k: {optimal_k}")
    print(f"Alpha: {optimal_alpha:.3f}")
    print(f"PSNR (Host vs Watermarked): {psnr_wm:.3f}")
    print(f"SSIM (Host vs Watermarked): {ssim_wm:.3f}")
    print(f"PSNR (Host vs Reconstructed): {psnr_rec:.3f}")
    print(f"SSIM (Host vs Reconstructed): {ssim_rec:.3f}")
    print(f"NC (Watermark vs Extracted Watermark): {nc:.3f}")

except Exception as e:
    print(f"Error in main execution: {e}")