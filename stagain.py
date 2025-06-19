import numpy as np
from scipy.integrate import quad

def rou(theta):
    def integrand(x, t):
        inner = np.sin(t) + np.sin(x)
        return np.sign(inner) * np.sqrt(np.abs(inner)) * np.sin(x)

    if np.isscalar(theta):
        result, _ = quad(lambda x: integrand(x, theta), 0, 2 * np.pi)
        return result
    else:
        return np.array([quad(lambda x: integrand(x, t), 0, 2 * np.pi)[0] for t in theta])

def fun_Y(keci, lamda, theta, x):
    pi = np.pi
    a1 = lamda ** 2
    a2 = 1 - 2 * keci * lamda

    A = 4 / pi * np.cos(theta) * (a1 + x) ** 0.5 / ((a2 - x) * x ** 2)
    B = 4 / pi * np.cos(theta) * np.sin(theta) * (a1 + x) / (a1 ** 0.5 * (a2 - x) * x ** 2)
    Y = A + B
    return Y

def find_x(keci, lamda, theta):
    a2 = 1 - 2 * keci * lamda
    n = 100
    x_arr = np.linspace(0.001, 0.99 * a2, n)  # 避免除以0
    Y1_min = np.inf
    x1_min = None

    for i in range(n):
        x = x_arr[i]
        try:
            Y1 = fun_Y(keci, lamda, theta, x)
            if Y1 < Y1_min:
                Y1_min = Y1
                x1_min = x
            else:
                break
        except Exception:
            continue  # 跳过非法输入

    return x1_min

def find_xtheta(keci, lamda):
    n = 100
    pi = np.pi
    theta_arr = np.linspace(0.001, 0.99 * pi / 2, n)
    Y2_max = -np.inf
    x2_min = None
    theta2_max = None

    for i in range(n):
        theta = theta_arr[i]
        x2 = find_x(keci, lamda, theta)
        Y2 = fun_Y(keci, lamda, theta, x2)
        if Y2 > Y2_max:
            Y2_max = Y2
            theta2_max = theta
            x2_min = x2
        else:
            break

    return [x2_min, theta2_max]

def fun_k11(keci, lamda, x, theta):
    pi = np.pi
    cn = -( (lamda - 2 * keci) ** 2 * x ** 2 + lamda ** 2 - 2 * lamda * x * (lamda - 2 * keci))
    cd = x * (lamda ** 2 + 2 * keci * lamda - 1) - lamda ** 2 * (1 - 2 * keci * lamda) + x ** 2
    c = cn / cd
    k11 = (4 * pi * c * np.cos(theta) / (rou(theta))**2) ** 0.5
    return k11

# 示例使用
keci = 1.0
lamda = 0.05

xtheta_sol = find_xtheta(keci, lamda)
x_sol = xtheta_sol[0]
theta_sol = xtheta_sol[1]

if x_sol is not None and theta_sol is not None:
    k11_sol = fun_k11(keci, lamda, x_sol, theta_sol)
    print("x =", x_sol)
    print("theta =", theta_sol)
    print("k11 =", k11_sol)
else:
    print("未找到有效的 x 或 theta 值，无法计算 k11")
