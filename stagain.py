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

def fun_J(keci, lamda, theta, x, beta):
    pi = np.pi
    a1 = lamda ** 2
    a2 = 1 - 2 * keci * lamda

    A = 4 / pi * np.cos(theta) * (a1 + x) ** 0.5 / ((a2 - x) * x ** 2)
    B = 4 / pi * np.cos(theta) * np.sin(theta) * (a1 + x) / (a1 ** 0.5 * (a2 - x) * x ** 2)
    J = beta * A + (1 - beta) * B
    return J

def find_x(keci, lamda, theta, beta):
    a2 = 1 - 2 * keci * lamda
    n = 100
    x_arr = np.linspace(0.001, 0.99 * a2, n)  
    J1_min = np.inf
    x1_min = None

    for i in range(n):
        x = x_arr[i]
        J1 = fun_J(keci, lamda, theta, x, beta)
        if J1 < J1_min:
                J1_min = J1
                x1_min = x
        else:
            break    
        
            
    return x1_min

def find_xtheta(keci, lamda, beta):
    n = 100
    pi = np.pi
    theta_arr = np.linspace(0.001, 0.99 * pi / 2, n)
    J2_max = -np.inf
    x2_min = None
    theta2_max = None

    for i in range(n):
        theta = theta_arr[i]
        x2 = find_x(keci, lamda, theta, beta)
        J2 = fun_J(keci, lamda, theta, x2, beta)
        if J2 > J2_max:
            J2_max = J2
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


#keci = 1.0
#lamda = 0.05
#beta=0.5
xtheta_sol = find_xtheta(keci, lamda, beta)
x_sol = xtheta_sol[0]
theta_sol = xtheta_sol[1]


k11_sol = fun_k11(keci, lamda, x_sol, theta_sol)
print("x =", x_sol)
print("theta =", theta_sol)
print("k11 =", k11_sol)

