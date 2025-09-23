import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# --- parâmetros / caminho ---
caminho = r"C:\Users\Hézio\Downloads\pendulo pasco-longo(in).csv"

# --- lê CSV ---
df = pd.read_csv(caminho, sep=";", decimal=",", encoding='utf-8')
t = df[df.columns[0]].to_numpy()
angle_raw = df[df.columns[1]].to_numpy()

# --- heurística: detectar se já está em graus ---
# se os valores absolutos máximos > 2π, assumimos que já estão em graus (ou em deg grande)
if np.nanmax(np.abs(angle_raw)) > 2*np.pi:
    ang = angle_raw.astype(float)
    print("Assumindo que os dados já estão em GRAUS (não converti).")
else:
    ang = np.degrees(angle_raw.astype(float))
    print("Convertei de rad -> graus.")

# --- limitar até 30 s ---
mask = t <= 30
t_fit = t[mask]
ang_fit = ang[mask]

# --- normaliza baseline estimada (apenas para estimativas iniciais) ---
if len(ang_fit) > 50:
    offset0 = np.median(ang_fit[-50:])
else:
    offset0 = np.median(ang_fit)
ang_zero = ang_fit - offset0

# --- estimativa de amplitude e frequência por FFT (chute inicial) ---
A0 = (np.max(ang_fit) - np.min(ang_fit)) / 2
dt = np.median(np.diff(t_fit))
if dt <= 0:
    dt = 1.0
N = len(t_fit)
# FFT para estimar frequência dominante
yf = np.fft.rfft(ang_zero - np.mean(ang_zero))
xf = np.fft.rfftfreq(N, dt)
# evitar componente zero
if len(xf) > 1:
    idx = np.argmax(np.abs(yf)[1:]) + 1
    f0 = xf[idx]
    omega0 = 2*np.pi * f0
else:
    omega0 = 2*np.pi * 0.5  # fallback
beta0 = 0.05
delta0 = 0.0

# --- modelo livre (mais robusto): A * exp(-beta t) * cos(omega t + delta) + offset ---
def damped_model(t, A, beta, omega, delta, offset):
    return A * np.exp(-beta * t) * np.cos(omega * t + delta) + offset

p0 = [A0, beta0, omega0, delta0, offset0]
# limites: beta >= 0, omega > 0
bounds = ([-np.inf, 0.0, 0.0, -10*np.pi, -np.inf],
          [ np.inf, 5.0, 1000.0,  10*np.pi,  np.inf])

# --- ajuste não-linear ---
params, cov = curve_fit(damped_model, t_fit, ang_fit, p0=p0, bounds=bounds, maxfev=20000)
A_fit, beta_fit, omega_fit, delta_fit, offset_fit = params
errs = np.sqrt(np.diag(cov))

print("\nParâmetros do fit (com erros):")
print(f"A     = {A_fit:.3f} ± {errs[0]:.3f}  (graus)")
print(f"beta  = {beta_fit:.5f} ± {errs[1]:.5f}  (1/s)")
print(f"omega = {omega_fit:.5f} ± {errs[2]:.5f}  (rad/s)  -> f = {omega_fit/(2*np.pi):.4f} Hz")
print(f"delta = {np.degrees(delta_fit):.3f} ± {np.degrees(errs[3]):.3f} (graus)")
print(f"offset= {offset_fit:.3f} ± {errs[4]:.3f} (graus)")

# --- visualizar ajuste ---
t_model = np.linspace(0, 30, 2000)
y_model = damped_model(t_model, *params)

plt.figure(figsize=(10,6))
plt.scatter(t_fit, ang_fit, s=12, color='red', label='Dados (até 30 s)')
plt.plot(t_model, y_model, color='blue', lw=2, label='Fit: A e^{-βt} cos(ωt+δ) + offset')
plt.xlabel("Tempo (s)"); plt.ylabel("Ângulo (graus)")
plt.title("Fit do pêndulo (0-30 s)")
plt.legend(); plt.grid(True)
plt.show()

# --- MÉTODO ALTERNATIVO (envelope linearizado) para checar β ---
# pega picos do valor absoluto do sinal sem offset
prom = max( (np.max(np.abs(ang_zero))*0.05, 1e-6) )
peak_idxs, _ = find_peaks(np.abs(ang_zero), prominence=prom)
if len(peak_idxs) >= 3:
    t_peaks = t_fit[peak_idxs]
    amp_peaks = np.abs(ang_zero[peak_idxs])
    maskp = amp_peaks > 0
    slope, intercept = np.polyfit(t_peaks[maskp], np.log(amp_peaks[maskp]), 1)
    beta_env = -slope
    A_env = np.exp(intercept)
    print(f"\nEnvelope (linearized) -> beta_env = {beta_env:.5f} 1/s, A_env ~ {A_env:.3f} deg")
else:
    print("\nPoucos picos detectados para o método do envelope (necessário >=3).")
