import numpy as np
from autograd import grad


def gradiente_simple(
    f, u_0, f_0, lambdaaaaa, learning_rate=0.1, max_iter=1_000, e=1e-4
):
    f_grad = grad(f)
    for i in range(max_iter):
        grad_u = f_grad(u_0, f_0, lambdaaaaa)
        u_1 = u_0 - learning_rate * grad_u
        diff_norm = np.linalg.norm(u_1 - u_0)
        print(
            f"Iteración {i}, ||u_1 - u_0|| = {diff_norm:.6f}, J(u) = {f(u_1, f_0, lambdaaaaa):.4f}"
        )
        if diff_norm <= e:
            print("x_1 - x_0 es casi cero")
            return u_1
        u_0 = u_1.copy()
    print(f"{max_iter} iteraciones alcanzadas")
    return u_1


def gradiente_momentum(
    f, u_0, f_0, lambdaaaaa, learning_rate=1e-6, momentum=0.8, max_iter=1_000, e=1e-4
):
    f_grad = grad(f)
    v_0 = np.zeros_like(u_0)

    for i in range(max_iter):
        grad_u = f_grad(u_0, f_0, lambdaaaaa)

        # Actualización de velocidad (momentum)
        v_1 = momentum * v_0 + learning_rate * grad_u

        # Actualización de u
        u_1 = u_0 - v_1
        diff_norm = np.linalg.norm(u_1 - u_0)

        print(
            f"Iteración {i}, ||u_1 - u_0|| = {diff_norm:.6f}, J(u) = {f(u_1, f_0, lambdaaaaa):.4f}"
        )

        if diff_norm <= e:
            print("x_1 - x_0 es casi cero")
            return u_1

        u_0, v_0 = u_1.copy(), v_1.copy()

    print(f"{max_iter} iteraciones alcanzadas")
    return u_1


def gradiente_nesterov(
    f, u_0, f_0, lambdaaaaa, learning_rate=1e-4, momentum=0.8, max_iter=1_000, e=1e-4
):
    f_grad = grad(f)
    v_0 = np.zeros_like(u_0)

    for i in range(max_iter):
        # Estimación adelantada
        u_adelantado = u_0 + momentum * v_0
        grad_u = f_grad(u_adelantado, f_0, lambdaaaaa)

        # Actualización de velocidad
        v_1 = momentum * v_0 - learning_rate * grad_u

        # Paso de actualización
        u_1 = u_0 + v_1
        diff_norm = np.linalg.norm(u_1 - u_0)

        print(
            f"Iteración {i}, ||u_1 - u_0|| = {diff_norm:.6f}, J(u) = {f(u_1, f_0, lambdaaaaa):.4f}"
        )

        if diff_norm <= e:
            print("x_1 - x_0 es casi cero")
            return u_1

        u_0, v_0 = u_1.copy(), v_1.copy()

    print(f"{max_iter} iteraciones alcanzadas")
    return u_1