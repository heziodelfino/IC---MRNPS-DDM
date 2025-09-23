import numpy as np
import matplotlib.pyplot as plt

# parâmetros físicos
g = 9.81   # gravidade local (m/s^2)
l = 0.9    # comprimento da haste (m)
beta = 0.05  # amortecimento (1/s)
dt = 0.001   # passo de tempo
Tmax = 30.0  # tempo total de simulação

# número de passos
N = int(Tmax/dt)

# arrays
t = np.linspace(0, Tmax, N)
theta = np.zeros(N)
omega = np.zeros(N)

# condições iniciais
theta[0] = np.radians(-150)  # posição inicial (rad)
omega[0] = 0.0               # velocidade angular inicial (rad/s)

# integração por Euler explícito
for i in range(N-1):
    f_i = -2*beta*omega[i] + (g/l)*np.sin(theta[i])  # aceleração angular
    omega[i+1] = omega[i] + f_i*dt
    theta[i+1] = theta[i] + omega[i]*dt

# converte para graus para plot
theta_deg = np.degrees(theta)

# gráfico
plt.figure(figsize=(10,6))
plt.plot(t, theta_deg, label="Método de Euler")
plt.scatter(t, theta_deg)
plt.xlabel("Tempo (s)")
plt.ylabel("Ângulo (graus)")
plt.title("Pêndulo amortecido não linear (Euler)")
plt.legend()
plt.grid(True)
plt.show()
