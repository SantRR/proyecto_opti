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


def gradiente_simple(f, u_0, f_0, lambdaaaaa, learning_rate, max_iter=10_000, e=1e-5):
    f_grad = grad(f)
    for _ in range(max_iter):
        grad_u = f_grad(u_0, f_0, lambdaaaaa)
        u_1 = u_0 - learning_rate * grad_u
        if np.linalg.norm(u_1 - u_0) <= e:
            print("x_1 - x_0 es casi cero")
            return u_1
        u_0 = u_1.copy()
    print(f"{max_iter} iteraciones alcanzadas")
    return u_1


def gradiente_momentum(
    f, u_0, f_0, lambdaaaaa, learning_rate, momentum=0.9, max_iter=10_000, e=1e-5
):
    f_grad = grad(f)
    v_0 = np.zeros_like(u_0)
    for _ in range(max_iter):
        grad_u = f_grad(u_0, f_0, lambdaaaaa)
        v_1 = momentum * v_0 + learning_rate * grad_u
        u_1 = u_0 - v_1
        if np.linalg.norm(u_1 - u_0) <= e:
            print("x_1 - x_0 es casi cero")
            return u_1
        u_0, v_0 = u_1.copy(), v_1.copy()
    print(f"{max_iter} iteraciones alcanzadas")
    return u_1


def gradiente_nesterov(
    f, u_0, f_0, lambdaaaaa, learning_rate, momentum=0.9, max_iter=10_000, e=1e-5
):
    f_grad = grad(f)
    v_0 = np.zeros_like(u_0)
    for _ in range(max_iter):
        u_adelantado = u_0 - momentum * v_0
        grad_u = f_grad(u_adelantado, f_0, lambdaaaaa)
        v_1 = momentum * v_0 + learning_rate * grad_u
        u_1 = u_0 - v_1
        if np.linalg.norm(u_1 - u_0) <= e:
            print("x_1 - x_0 es casi cero")
            return u_1
        u_0, v_0 = u_1.copy(), v_1.copy()
    print(f"{max_iter} iteraciones alcanzadas")
    return u_1
