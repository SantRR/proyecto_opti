import time

import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def evaluar_resultado(
    imagen_denoiseada, imagen_referencia, nombre_metodo, std_ruido, tiempo_ejecucion
):
    psnr_val = psnr(imagen_referencia, imagen_denoiseada, data_range=255)
    ssim_val = ssim(imagen_referencia, imagen_denoiseada, data_range=255)
    print(f"\nResultados - Método: {nombre_metodo}, Ruido STD: {std_ruido}")
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"Tiempo de ejecución: {tiempo_ejecucion:.4f} segundos")


def mostrar_resultado(original, noisy, denoised, metodo, std):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(
        axs,
        [original, noisy, denoised],
        ["Original", f"Ruidosa (std={std})", f"Denoised ({metodo})"],
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def medir_tiempo(funcion, *args, **kwargs):
    inicio = time.time()
    resultado = funcion(*args, **kwargs)
    fin = time.time()
    return resultado, fin - inicio