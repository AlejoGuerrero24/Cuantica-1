import numpy as np
import matplotlib.pyplot as plt

a = 1  # Valor de 'a' en la función de onda
A = (30/(a**3))**1/2  # Valor de 'A' en la función de onda
n_valores = [1, 3, 5]  # Valores de 'n' para graficar

def Onda(x, n):
    return A * x * (a - x) * np.sin(n * np.pi * x / a)

def Probabilidad(x, n):
    return (Onda(x, n))**2

x = np.linspace(0, a, 1000)

fig, axs = plt.subplots(len(n_valores), 2, figsize=(10, 8))

for i, n in enumerate(n_valores):
    psi = Onda(x, n)
    prob = Probabilidad(x, n)

    axs[i, 0].plot(x, psi, label=f'n = {n}')
    axs[i, 0].set_title(f'Función de onda para n = {n}')
    axs[i, 0].set_xlabel('x')
    axs[i, 0].set_ylabel('Psi(x)')

    axs[i, 1].plot(x, prob, label=f'n = {n}')
    axs[i, 1].set_title(f'Distribución de probabilidad para n = {n}')
    axs[i, 1].set_xlabel('x')
    axs[i, 1].set_ylabel('|Psi(x)|^2')

plt.tight_layout()
plt.legend()
plt.show()
