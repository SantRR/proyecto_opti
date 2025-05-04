import time
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def evaluar_resultado(imagen_original, imagen_denoised):
    valor_psnr = psnr(imagen_original, imagen_denoised, data_range=255)
    valor_ssim = ssim(imagen_original, imagen_denoised, data_range=255)
    return valor_psnr, valor_ssim


def mostrar_resultado(original, noisy, denoised, metodo, std):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axs, [original, noisy, denoised],
                               ["Original", f"Ruidosa (std={std})", f"Denoised ({metodo})"]):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def medir_tiempo(funcion, *args, **kwargs):
    inicio = time.time()
    resultado = funcion(*args, **kwargs)
    fin = time.time()
    return resultado, fin - inicio
