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
# restringe ao intervalo 50–150 segundos
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
plt.scatter(t_peaks, y_log, edgecolors='#FF3430', label='ln(Δθ_pico) (dados)', 
            marker='o', s=40, facecolors='none')

t_line = np.linspace(np.min(t_peaks), np.max(t_peaks), 200)
print(np.min(t_peaks), np.max(t_peaks))

# Cria string formatada da equação da reta
eq_label = (r"$\ln(\Delta\theta) = ({:.5f})t + ({:.5f})$"
            "\n"
            r"$\beta = {:.5f}\ \mathrm{{s}}^{{-1}},\ R = {:.4f}$"
            .format(slope, intercept, -slope, r_value))

plt.plot(t_line, slope * t_line + intercept, color="#272727",
         label=eq_label, lw=3, linestyle='--')

plt.xlabel("Tempo (s)", fontsize=16)
plt.ylabel(r"$\ln(\Delta\theta_{\mathrm{pico}})$", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.axvspan(150, 1500, color="#FF8330", alpha=0.20, label="2° Regime")
plt.title("Envelope linearizado (150–1500 s)", fontsize  = 16)
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------------------------------
# FIM
# ---------------------------------------------------------------------------