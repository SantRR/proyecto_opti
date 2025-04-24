import numpy as np


def gradiente_newton(grad, hess, x_0, max_iter=10_000, e=1e-5):
    for _ in range(max_iter):
        x_1 = x_0 - np.linalg.inv(hess(x_0)) @ grad(x_0)
        if np.linalg.norm(x_1 - x_0) <= e:
            print("x_1 - x_0 es casi cero")
            return x_1
        x_0 = x_1.copy()
    print("{max_iter} iteraciones alcanzadas")
    return x_1


def hessiano(vec):
    return np.array(
        [[2 - 400 * vec[1] + 1200 * vec[0] ** 2, -400 * vec[0]], [-400 * vec[0], 200]]
    )


def gradiente(vec):
    return np.array(
        [
            2 * vec[0] - 2 - 400 * vec[0] * vec[1] + 400 * vec[0] ** 3,
            200 * vec[1] - 200 * vec[0] ** 2,
        ]
    ).T


x_0 = np.array([2, 4])
res = gradiente_newton(gradiente, hessiano, x_0)
print(res)
