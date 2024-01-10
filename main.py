import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

G = 6.674 * 10 ** -11

def u(x):
    return 5 - x / 3


def rho(x):
    return 0 if x <= 1 or x > 2 else 1

# Shape function ei
def e(i, x, h):
    if x < h * (i - 1) or x > h * (i + 1):
        return 0
    elif x <= h * i:
        return (x - h * (i - 1)) / h
    else:
        return (h * (i + 1) - x) / h

# And its derivative
def e_prim(i, x, h):
    if x < h * (i - 1) or x > h * (i + 1):
        return 0
    elif x <= h * i:
        return 1 / h
    else:
        return -1 / h


# B(ei, ej)
def B(i, j, h):
    s, z = h * (i - 1), h * (i + 1)  # We calculate domains of functions where they are not zero
    if j is None:  # It is used to calculate L(ei)
        return 1 / 3 * (e(i, z, h) - e(i, s, h))  # We don't need to calculate the integral for B(ei, u) because u' is -1/3
    else:
        return -quad(lambda x: e_prim(i, x, h) * e_prim(j, x, h), s, z)[0]


# L(ei)
def L(i, h):
    s, z = h * (i - 1), h * (i + 1)
    return quad(lambda x: 4 * np.pi * G * rho(x) * e(i, x, h), s, z)[0] - B(i, None, h)


def create_matrices_optimized(n, h):
    A = np.zeros((n-2, n-2))  # We calculate B(ei, ej) and L(ei) for 1<=i,j<=n-1 because w(0)=w(3)=0
    b = np.zeros(n-2)
    A[0, 0] = A[-1, -1] = B(1, 1, h)
    for i in range(0, n - 3):
        A[i, i + 1] = B(i+1, i + 2, h)
        A[i + 1, i] = A[i, i + 1]  # B(ei, ej) = B(ej, ei) and if |i-j|>1, then B(ei, ej) = 0
        A[i, i] = A[0, 0]  # B(ei, ei) = B(e1, e1) for all i
        b[i] = L(i+1, h)

    return A, b


def show(x, y):
    plt.figure(figsize=(10, 7))
    plt.plot(x, y, color='red')
    plt.title('Gravitational Potential FEM Solution')
    plt.xlabel('x')
    plt.ylabel('φ(x)')
    plt.grid(True)
    plt.show()
    print('u(0)=' + str(y[0]), 'u(3)=' + str(y[-1]))


def solve_optimized(n):
    start = time.time()
    h = 3 / (n-2)
    A, b = create_matrices_optimized(n, h)
    y = np.linalg.solve(A, b)
    x = np.linspace(h, 3-h, n-2)  # We will add 0 and 3 later
    u_values = [u(xi) for xi in x]
    y = y + u_values
    print('Time needed to solve the equation was: ' + str(round(time.time() - start, 2)) + ' s')
    show([0] + list(x) + [3], [5] + list(y) + [4])  # We need to add boundary values


n = int(input('Enter the value of n: '))
solve_optimized(n)
