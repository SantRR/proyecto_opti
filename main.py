from ruido import *

if __name__ == "__main__":
    input_path = "imagen_original.png"
    output_path = "imagen_con_ruido.png"

    image = load_image_grayscale(input_path)
    noisy = add_gaussian_noise(image, mean=0, std=20)
    save_image(output_path, noisy)
    print(f"Imagen guardada con ruido en: {output_path}")
