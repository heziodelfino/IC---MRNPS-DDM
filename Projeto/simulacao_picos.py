import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# caminho do CSV 
# ---------------------------------------------------------------------------
caminho = r"C:\Users\Hézio\Downloads\pendulo pasco-longo(in).csv"

# ---------------------------------------------------------------------------
# lê os dados
# ---------------------------------------------------------------------------
df = pd.read_csv(caminho, sep=";", decimal=",", encoding="utf-8")
t = df.iloc[:, 0].to_numpy(dtype=float)
ang_raw = df.iloc[:, 1].to_numpy(dtype=float)

# ---------------------------------------------------------------------------
# converte para graus se estiver em rad (heurística)
# ---------------------------------------------------------------------------
if np.nanmax(np.abs(ang_raw)) > 2 * np.pi:
    ang_deg = ang_raw.copy()
    print("Assumindo dados já em graus (não converti).")
else:
    ang_deg = np.degrees(ang_raw)
    print("Convertemos rad -> graus.")

# ---------------------------------------------------------------------------
# restringe ao intervalo 150–1500 segundos
# ---------------------------------------------------------------------------
mask = (t >= 50) & (t <= 150)
t_win = t[mask]
ang_win = ang_deg[mask]

if len(t_win) < 10:
    raise RuntimeError("Poucos pontos no intervalo 150–1500 s — verifique o arquivo ou o corte de tempo.")

# ---------------------------------------------------------------------------
# estimativa de offset (baseline) -> subtrair antes de extrair os picos
# ---------------------------------------------------------------------------
offset0 = np.median(ang_win)
signal = ang_win - offset0  # sinal centrado

# ---------------------------------------------------------------------------
# estima período dominante por FFT (para escolher 'distance' em find_peaks)
# ---------------------------------------------------------------------------
dt = np.median(np.diff(t_win))
if dt <= 0 or np.isnan(dt):
    dt = 1.0

N = len(t_win)
yf = np.fft.rfft(signal - np.mean(signal))
xf = np.fft.rfftfreq(N, dt)
if len(xf) > 1:
    idx = np.argmax(np.abs(yf)[1:]) + 1
    f0 = xf[idx]
    period_est = 1.0 / f0 if f0 > 0 else None
else:
    period_est = None

if period_est is not None:
    distance_samples = max(3, int(0.6 * period_est / dt))
else:
    distance_samples = 10

# ---------------------------------------------------------------------------
# detecta apenas picos positivos
# ---------------------------------------------------------------------------
prominence = max((np.max(signal) - np.min(signal)) * 0.05, 1e-3)
peaks_pos, _ = find_peaks(signal, distance=distance_samples, prominence=prominence)

t_peaks = t_win[peaks_pos]
theta_peaks = signal[peaks_pos]          # já centrados (offset removido)

# ---------------------------------------------------------------------------
# Δθ_pico = |θ_pico - offset|
# ---------------------------------------------------------------------------
delta_theta_peaks = np.abs(theta_peaks)

# filtra valores pequenos
valid = delta_theta_peaks > 1e-6
t_peaks = t_peaks[valid]
delta_theta_peaks = delta_theta_peaks[valid]
theta_peaks = theta_peaks[valid]

if len(delta_theta_peaks) < 3:
    print("AVISO: menos que 3 picos válidos detectados — resultados podem não ser confiáveis.")

# ---------------------------------------------------------------------------
# linearização e ajuste linear
# ln(Δθ) = ln(A) - beta * t
# ---------------------------------------------------------------------------
y_log = np.log(delta_theta_peaks)
slope, intercept, r_value, p_value, std_err = linregress(t_peaks, y_log)
beta_env = -slope

# cálculo de incertezas
n = len(y_log)
y_fit_line = slope * t_peaks + intercept
residuals = y_log - y_fit_line
if n > 2:
    Sxx = np.sum((t_peaks - np.mean(t_peaks))**2)
    s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
    se_slope = s_err / np.sqrt(Sxx)
    se_intercept = s_err * np.sqrt(np.sum(t_peaks**2) / (n * Sxx))
else:
    se_slope = std_err
    se_intercept = np.nan

beta_err = se_slope
A_env = np.exp(intercept)
A_env_err = A_env * se_intercept if not np.isnan(se_intercept) else np.nan

# ---------------------------------------------------------------------------
# imprime resultados
# ---------------------------------------------------------------------------
print("\nRESULTADOS (50–150 s):")
print(f"n_picos usados = {n}")
print(f"beta_env = {beta_env:.5f} ± {beta_err:.5f} 1/s")
print(f"A_env    = {A_env:.3f} ± {A_env_err:.3f} graus")
print(f"R (pearson) = {r_value:.4f}")

# ---------------------------------------------------------------------------
# plota: ln(Δθ_pico) vs t com a reta ajustada
# ---------------------------------------------------------------------------
plt.figure(figsize=(9,6))
plt.scatter(t_peaks, y_log, edgecolors='#FF3430', label='ln(Δθ_pico) (dados)', marker='o', s=40, facecolors = 'none')
t_line = np.linspace(np.min(t_peaks), np.max(t_peaks), 200)
plt.plot(t_line, slope * t_line + intercept, color="#272727",
         label=f'Ajuste linear: slope={slope:.5f} -> β={-slope:.5f} 1/s\nR={r_value:.4f}', lw=3, linestyle = '--')
plt.xlabel("Tempo (s)", fontsize = 16)
plt.ylabel("ln(Δθ_pico)", fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.axvspan(50, 150, color='#A530FF', alpha=0.15, label="1° Regime")
plt.title("Envelope linearizado (50–150 s)")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------------------------------
# Definindo a função damped_model
# ---------------------------------------------------------------------------
def damped_model(t, A, beta, omega, delta, offset):
    return A * np.exp(-beta * t) * np.cos(omega * t + delta) + offset

# ---------------------------------------------------------------------------
# opcional: plota sinal original (150–1500s) com as assíntotas teóricas
# ---------------------------------------------------------------------------
t_model = np.linspace(50, 150, 10000)
env = 75.539 * np.exp(-0.00336 * t_model)
env_2 = 82.539 * np.exp(-0.00336 * t_model)
upper_env = env + offset0
lower_env = -env_2 + offset0

plt.figure(figsize=(10,6))
plt.plot(t_win, ang_win, 'p', ms=2, label='Dados (150–1500 s)', color = "#272727")
plt.plot(t_model, upper_env, lw=3, zorder=3,
         label=r'$\theta(t) = +75.539 \, e^{-0.00336 t} + 0.0$', color = "#90D136", linestyle = '--')
plt.plot(t_model, lower_env, lw=3, zorder=3,
         label=r'$\theta(t) = -82.539  \, e^{-0.00336 t} + 0.0$', color = "#6DFF7E", linestyle= '--')

# ---------------------------------------------------------------------------
# Ajuste com beta fixo (usando o modelo amortecido)
# ---------------------------------------------------------------------------
# Definindo beta fixo (do envelope)
beta_fix = 0.00336 

# Parâmetros aproximados:
A_fix = 75.539          
omega_fix = 3.034 
delta_fix = 0.0
offset_fix = -81.245



# Calcula curva teórica com beta fixo
y_model_fix = damped_model(t_model, A_fix, beta_fix, omega_fix, delta_fix, offset_fix)

# Plota no mesmo gráfico
plt.plot(t_model, y_model_fix, '--', color='#3F7EFC', lw=1.2,
         label=f'Ajuste com β encontrado = {beta_fix:.5f} 1/s', alpha = 0.5)

plt.scatter(t_peaks, theta_peaks + offset0, s=40, facecolors='none',
            edgecolors = '#FF3430', label='picos usados')

plt.xlabel("Tempo (s)", fontsize = 16)
plt.ylabel("Ângulo (°)", fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.axvspan(50, 150, color='#A530FF', alpha=0.15, label="1° Regime")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------------------------------
# FIM
# ---------------------------------------------------------------------------
