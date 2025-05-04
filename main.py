import os

from evaluador import *
from gradientes import *
from optimizador import *
from ruido import *

if __name__ == "__main__":
    cwd = os.getcwd()
    path_imagen_original = cwd + "/imagen_original.png"
    image = load_image_grayscale(path_imagen_original)

    niveles_ruido = [10, 25, 50, 75]
    lambdaaaaa = 0.1
    learning_rate = 0.1
    max_iter = 300

    for std in niveles_ruido:
        print(f"\n====== Nivel de ruido STD={std} ======")
        noisy = add_gaussian_noise(image, std=std)
        save_image(f"{cwd}/imagen_ruido_{std}.png", noisy)

        f = noisy.copy()
        u_0 = noisy.copy()

        metodos = {
            "Gradiente Simple": gradiente_simple,
            "Momentum": gradiente_momentum,
            "Nesterov": gradiente_nesterov,
        }

        for nombre, metodo in metodos.items():
            print(f"\nMétodo: {nombre}")
            loss_func = lambda u: func_loss(u, f, lambdaaaaa)

            u_resultado, tiempo = medir_tiempo(
                metodo, loss_func, u_0.copy(), learning_rate, max_iter=max_iter
            )

            psnr_val, ssim_val = evaluar_resultado(image, u_resultado)
            print(
                f"Tiempo: {tiempo:.2f}s | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.3f}"
            )

            save_image(
                f"{cwd}/denoised_{nombre.replace(' ', '_')}_std{std}.png", u_resultado
            )

            mostrar_resultado(image, noisy, u_resultado, nombre, std)
