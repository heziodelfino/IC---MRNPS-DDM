import numpy as np
import matplotlib.pyplot as plt

"""
Paleta de cores :D
[MAGENTA] #9671bd #6a408d
[AZUL]    #378d94 #77b5b6 #205458
[VERDE]   #beee62 #70ae6e #375736
"""

# ===================== PARÂMETROS =====================
A = 0.2
omega0 = 1.7
phi0 = 0.0

gamma_sub = 0.01        
t_max = 20.0
N = 1000

noise_level = 0.05      #0.00 p/ sem ruído
seed = 0

outfile = "pendulo_subcritico_gamma_001.npz"
# ===================================================================

#domínio de t
t = np.linspace(0, t_max, N)

# checa se tá no regime subcrítico
if not (0 <= gamma_sub < omega0):
    raise ValueError(f"gamma_sub precisa satisfazer 0 <= gamma_sub < omega0. "
                     f"Recebi gamma_sub={gamma_sub}, omega0={omega0}")

omega_d = np.sqrt(omega0**2 - gamma_sub**2)

# sinal limpo
x_clean = A * np.exp(-gamma_sub * t) * np.cos(omega_d * t + phi0)

# com ruído
rng = np.random.default_rng(seed)
x_noisy = x_clean + noise_level * rng.normal(size=len(t))

# velocidades (pra espaço de fase)
omega_clean = np.gradient(x_clean, t)
omega_noisy = np.gradient(x_noisy, t)

# ===================== checagem visual =====================
plt.figure(figsize=(10,5))
plt.plot(t, x_clean, lw=2, label="clean")
plt.scatter(t, x_noisy, s=15, alpha=0.6, label="noisy")
plt.xlabel("t (s)")
plt.ylabel(r"$\theta(t)$ (rad)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ===================== SALVA O DATASET =====================
np.savez(
    outfile,
    t=t,
    x_clean=x_clean,
    x_noisy=x_noisy,
    omega_clean=omega_clean,
    omega_noisy=omega_noisy,
    A=A,
    omega0=omega0,
    omega_d=omega_d,
    gamma=gamma_sub,
    phi0=phi0,
    noise_level=noise_level,
    seed=seed
)

print(f"Salvo: {outfile}")
