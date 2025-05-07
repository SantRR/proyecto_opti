import numpy as np
from autograd import grad
from optimizador import grad_func_objetivo

def descenso_gradiente_simple(
    f, u_0, f_0, param_lambda, learning_rate=0.1, max_iter=1_000, e=1e-4
):
    f_grad = grad(f)
    for i in range(max_iter):
        grad_u = f_grad(u_0, f_0, param_lambda)
        u_1 = u_0 - learning_rate * grad_u
        diff_norm = np.linalg.norm(u_1 - u_0)
        print(
            f"Iteración {i}, ||u_1 - u_0|| = {diff_norm:.6f}, J(u) = {f(u_1, f_0, param_lambda):.4f}"
        )
        if diff_norm <= e:
            print("x_1 - x_0 es casi cero")
            return u_1
        u_0 = u_1.copy()
    print(f"{max_iter} iteraciones alcanzadas")
    return u_1


def descenso_gradiente_momentum(u, f, lambda_param, learning_rate, momentum, num_iter, tol=1e-6):
    v = np.zeros_like(u)  
    prev_u = u.copy()      
    for i in range(num_iter):
        gradiente = grad_func_objetivo(u, f, lambda_param)
        v = momentum * v + (1 - momentum) * gradiente  # Momentum
        u = u - learning_rate * v  

        diff = np.linalg.norm(u - prev_u)
        print(f"Iteración {i+1}, Diferencia: {diff}")
        if diff < tol:
            print(f"Convergencia alcanzada en la iteración {i+1}")
            break
        prev_u = u.copy()
    
    return u



def descenso_gradiente_nesterov(u, f, lambda_param, learning_rate, momentum, num_iter, tol=1e-6):
    v = np.zeros_like(u)  
    prev_u = u.copy()  
    for i in range(num_iter):
        u_temp = u - momentum * v 
        gradiente = grad_func_objetivo(u_temp, f, lambda_param)
        v = momentum * v + (1 - momentum) * gradiente  
        u = u - learning_rate * v  

        diff = np.linalg.norm(u - prev_u)
        print(f"Iteración {i+1}, Diferencia: {diff}")
        if diff < tol:
            print(f"Convergencia alcanzada en la iteración {i+1}")
            break
        prev_u = u.copy()
    
    return u