import numpy as np
from autograd import grad, hessian


def gradiente_newton(f, x_0, max_iter=10_000, e=1e-5):
    f_grad, f_hessian = grad(f), hessian(f)
    for _ in range(max_iter):
        x_1 = x_0 - np.linalg.inv(f_hessian(x_0)) @ f_grad(x_0)
        if np.linalg.norm(x_1 - x_0) <= e:
            print("x_1 - x_0 es casi cero")
            return x_1
        x_0 = x_1.copy()
    print(f"{max_iter} iteraciones alcanzadas")
    return x_1


def gradiente():
    pass


def gradiente_momentum():
    pass
