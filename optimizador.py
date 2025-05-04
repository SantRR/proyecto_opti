import autograd.numpy as np
from autograd import grad


def grad_x(u):
    return u[1:, :] - u[:-1, :]


def grad_y(u):
    return u[:, 1:] - u[:, :-1]


def grad_norma_cuadrada(u):
    gx = grad_x(u)
    gy = grad_y(u)
    return np.sum(gx**2) + np.sum(gy**2)


def func_loss(u, f, lambdaaaaa):
    return 0.5 * np.sum((u - f) ** 2) + 0.5 * lambdaaaaa * grad_norma_cuadrada(u)


grad_loss = grad(func_loss, 0)
