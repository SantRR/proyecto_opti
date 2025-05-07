import autograd.numpy as np
from autograd import grad


def func_objetivo(f, u, param_lambda):
    grad_u = np.gradient(u)
    grad_u = np.array(grad_u)  # Convierte a arreglo para hace operaciones vectoriales
    return 0.5 * np.sum((u - f)**2) + 0.5 * param_lambda * np.sum(grad_u**2)

grad_func_objetivo = grad(func_objetivo)
