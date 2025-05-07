import os

import autograd.numpy as np
from evaluador import *
from gradientes import *
from optimizador import func_loss
from ruido import *

if __name__ == "__main__":
    cwd = os.getcwd()
    path_imagen_original = os.path.join(cwd, "imagen_original.png")
    path_imagen_con_ruido = os.path.join(cwd, "imagen_con_ruido.png")

    imagen_original = load_image_grayscale(path_imagen_original)

    niveles_ruido = [10, 30, 75]
    max_iter = 100
    e = 1e-4
    learning_rate = 0.1

    #"""
    metodos = {
        "Gradiente Simple": gradiente_simple,
        "Momentum": gradiente_momentum,
        "Nesterov": gradiente_nesterov,
    }
    #"""

    """
    metodos = {
        "Momentum": gradiente_momentum,
        "Nesterov": gradiente_nesterov,
    }
    """

    for std in niveles_ruido:
        print(f"\n====== Nivel de ruido STD={std} ======\n")
        imagen_con_ruido = add_gaussian_noise(imagen_original, std=std).astype(
            np.float32
        )
        save_image(path_imagen_con_ruido, imagen_con_ruido)
        std_lambda = 0.2

        for nombre, metodo in metodos.items():
            print(f"\nMétodo: {nombre}")

            u_resultado, tiempo = medir_tiempo(
                metodo,
                func_loss,
                imagen_con_ruido.copy(),
                imagen_con_ruido.copy(),
                std_lambda,
                learning_rate,
                max_iter,
            )
            evaluar_resultado(u_resultado, imagen_original, nombre, std, tiempo)
            nombre_archivo_salida = (
                f"denoised_{nombre.replace(' ', '_').lower()}_std{std}.png"
            )
            path_salida = os.path.join(cwd, nombre_archivo_salida)
            save_image(path_salida, u_resultado.astype(np.uint8))