"""
Fit com beta fixo e parâmetros iniciais fornecidos manualmente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# ----------------------------------------------------------------------------------------------------
# Caminho + intervalo de tempo
# ----------------------------------------------------------------------------------------------------
caminho = r"C:\Users\Hézio\Downloads\pendulo pasco-longo(in).csv"
t_min, t_max = 150, 1500

# ----------------------------------------------------------------------------------------------------
# Coeficientes conhecidos (tirados do fit da eq.(20))
# ----------------------------------------------------------------------------------------------------
a1 = 7.458815e-03
a2 = 4.611150e-05
T0 = 2.241572
beta_known = 0.00125

# ----------------------------------------------------------------------------------------------------
# Parâmetros iniciais 
# ----------------------------------------------------------------------------------------------------
A_init = -50
delta_init = 31.41
offset_init = -83.31

# ----------------------------------------------------------------------------------------------------
# Ler dados do caminho
# ----------------------------------------------------------------------------------------------------
df = pd.read_csv(caminho, sep=";", decimal=",", encoding='utf-8')
t = df[df.columns[0]].to_numpy()
angle_raw = df[df.columns[1]].to_numpy()

# Converter radianos para graus se necessário
if np.nanmax(np.abs(angle_raw)) > 2*np.pi:
    ang = angle_raw.astype(float)
    print("Assumindo que os dados já estão em GRAUS.")
else:
    ang = np.degrees(angle_raw.astype(float))
    print("Convertemos RAD -> GRAUS.")

# Selecionar janela de tempo (regimes)
mask = (t >= t_min) & (t <= t_max)
t_fit = t[mask]
ang_fit = ang[mask]

if len(t_fit) < 6:
    raise RuntimeError("Poucos pontos no intervalo selecionado. Ajuste t_min/t_max.")

# ----------------------------------------------------------------------------------------------------
# Modelo com beta fixo com base na eq. (23) do paper
# ----------------------------------------------------------------------------------------------------
def damped_model_variable_period_beta_fixed(t, A, delta, offset, beta=beta_known):
    """
    theta(t) = \theta_{0}e^{-\beta t}cos{(\frac{2 \pi t}{a_{2}\theta_{0}e^{-2 \beta t}+a_{1}\theta_{0}e^+{-\beta t} + T_{0}})}
    """
    T_t = a2 * A * np.exp(-2.0 * beta * t) + a1 * A * np.exp(-beta * t) + T0
    T_t = np.where(T_t <= 1e-9, 1e-9, T_t)  # isso evita divisão por zero
    omega_t = 2.0 * np.pi / T_t
    amplitude_t = A * np.exp(-beta * t)
    return amplitude_t * np.cos(omega_t * t + delta) + offset

# ----------------------------------------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------------------------------------
# Parâmetros iniciais manuais
p0 = [A_init, delta_init, offset_init]
# bounds bem amplos, só pra garantir
bounds = ([-500, -10*np.pi, -500],
          [ 500,  10*np.pi,  500])

params, cov = curve_fit(
    damped_model_variable_period_beta_fixed,
    t_fit, ang_fit,
    p0=p0, bounds=bounds, maxfev=40000
)

A_fit, delta_fit, offset_fit = params
errors = np.sqrt(np.diag(cov))

# ----------------------------------------------------------------------------------------------------
# Resultados
# ----------------------------------------------------------------------------------------------------
print("\n==== RESULTADOS DO FIT (BETA FIXO) ====")
print(f"A       = {A_fit:.6f} ± {errors[0]:.6f}")
print(f"beta    = {beta_known:.6f} (fixo)")
print(f"delta   = {delta_fit:.6f} rad  ({np.degrees(delta_fit):.6f}°)")
print(f"offset  = {offset_fit:.6f} ± {errors[2]:.6f}")
print("=======================================")

# ----------------------------------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------------------------------
t_model = np.linspace(np.min(t_fit), np.max(t_fit), 10000)
y_model = damped_model_variable_period_beta_fixed(t_model, A_fit, delta_fit, offset_fit)

plt.figure(figsize=(11,5))
plt.scatter(t_fit, ang_fit, s=8, c='k', label='dados')
plt.plot(t_model, y_model, '--', color='#3F7EFC', lw=1.5, alpha = 0.7)
plt.xlabel("Tempo (s)", fontsize = 16)
plt.ylabel("Ângulo (°)", fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.title("Fit do pêndulo com base na eq. (20) 2° regime (150–1500 s)", fontsize = 16)
plt.legend()
plt.grid(alpha=0.70)
plt.xlim(np.min(t_fit), np.max(t_fit))
plt.show()

# ----------------------------------------------------------------------------------------------------
# FIM 
# ----------------------------------------------------------------------------------------------------