import os

import autograd.numpy as np
from evaluador import *
from gradientes import *
from optimizador import grad_func_objetivo
from ruido import *

if __name__ == "__main__":
    cwd = os.getcwd()
    path_imagen_original = os.path.join(cwd, "imagen_original.png")
    path_imagen_con_ruido = os.path.join(cwd, "imagen_con_ruido.png")

    imagen_original = load_image_grayscale(path_imagen_original)

    niveles_ruido = [10, 30, 75]
    learning_rates = [0.1, 0.01, 0.001]
    max_iter = 100
    e = 1e-4
    param_lambda = 0.2

    #"""
    metodos = {
        "Gradiente Simple": descenso_gradiente_simple,
        "Momentum": descenso_gradiente_momentum,
        "Nesterov": descenso_gradiente_nesterov,
    }
    #"""

    """
    metodos = {
        "Momentum": gradiente_momentum,
        "Nesterov": gradiente_nesterov,
    }
    """

    for std in niveles_ruido:
        print(f"====== Nivel de ruido STD={std} ======\n")
        imagen_con_ruido = add_gaussian_noise(imagen_original, std=std).astype(np.float32)
        save_image(path_imagen_con_ruido, imagen_con_ruido)

        for lr in learning_rates:
            print(f"Probando con LR: {lr}")

            for nombre, metodo in metodos.items():
                print(f"\nMétodo: {nombre}")

                if nombre == "Momentum" or nombre == "Nesterov":
                    u_resultado, tiempo = medir_tiempo(
                        metodo,
                        imagen_con_ruido.copy(),
                        imagen_original,
                        param_lambda,
                        learning_rate=lr,
                        momentum=0.9,
                        num_iter=max_iter,
                        tol=e
                    )
                else:
                    u_resultado, tiempo = medir_tiempo(
                        metodo,
                        imagen_con_ruido.copy(),
                        imagen_original,
                        param_lambda,
                        learning_rate=lr,
                        max_iter=max_iter,
                        e=e
                    )

                evaluar_resultado(u_resultado, imagen_original, nombre, std, tiempo)
                nombre_archivo_salida = f"denoised_{nombre.replace(' ', '_').lower()}_lr{lr}_std{std}.png"
                path_salida = os.path.join(cwd, nombre_archivo_salida)
                save_image(path_salida, u_resultado.astype(np.uint8))
