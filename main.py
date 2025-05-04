import os
from gradientes import *
from optimizador import *
from ruido import *

if __name__ == "__main__":
    cwd = os.getcwd()
    path_imagen_original = os.path.join(cwd, "imagen_original.png")
    path_imagen_ruido = os.path.join(cwd, "imagen_con_ruido.png")

    image = load_image_grayscale(path_imagen_original)

    noisy = add_gaussian_noise(image, mean=0, std=75)
    save_image(path_imagen_ruido, noisy)
    print(f"Imagen guardada con ruido en: {path_imagen_ruido}")

    learning_rate = 0.1
    num_iterations = 100
    lambdaaaaa = 0.1

    f = image
    u = noisy

    u_simple = gradiente_simple(func_loss, u, f, lambdaaaaa, learning_rate, num_iterations)
    print("Pérdida final (GD):", func_loss(u_simple, f, lambdaaaaa))

    u_momentum = gradiente_momentum(func_loss, u, f, lambdaaaaa, learning_rate, num_iterations, momentum=0.9)
    print("Pérdida final (Momentum):", func_loss(u_momentum, f, lambdaaaaa))

    u_nesterov = gradiente_nesterov(func_loss, u, f, lambdaaaaa, learning_rate, num_iterations, momentum=0.9)
    print("Pérdida final (Nesterov):", func_loss(u_nesterov, f, lambdaaaaa))
