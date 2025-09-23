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
# restringe aos primeiros 30 segundos
# ---------------------------------------------------------------------------
mask30 = t <= 30
t30 = t[mask30]
ang30 = ang_deg[mask30]

if len(t30) < 10:
    raise RuntimeError("Poucos pontos nos primeiros 30 s — verifique o arquivo ou o corte de tempo.")

# ---------------------------------------------------------------------------
# estimativa de offset (baseline) -> subtrair antes de extrair os picos
# uso mediana para robustez
# ---------------------------------------------------------------------------
offset0 = np.median(ang30)
signal = ang30 - offset0  # sinal centrado

# ---------------------------------------------------------------------------
# estima período dominante por FFT (para escolher 'distance' em find_peaks)
# ---------------------------------------------------------------------------
dt = np.median(np.diff(t30))
if dt <= 0 or np.isnan(dt):
    dt = 1.0

N = len(t30)
yf = np.fft.rfft(signal - np.mean(signal))
xf = np.fft.rfftfreq(N, dt)
if len(xf) > 1:
    # pega máximo exceto componente DC
    idx = np.argmax(np.abs(yf)[1:]) + 1
    f0 = xf[idx]
    period_est = 1.0 / f0 if f0 > 0 else None
else:
    period_est = None

if period_est is not None:
    distance_samples = max(3, int(0.6 * period_est / dt))  # 0.6*period -> garante separar picos vizinhos
else:
    distance_samples = 10

# ---------------------------------------------------------------------------
# detecta picos positivos e negativos (no sinal centrado)
# ---------------------------------------------------------------------------
prominence = max((np.max(signal) - np.min(signal)) * 0.05, 1e-3)  # 5% do swing, mínimo
peaks_pos, _ = find_peaks(signal, distance=distance_samples, prominence=prominence)
peaks_neg, _ = find_peaks(-signal, distance=distance_samples, prominence=prominence)
pidx = np.sort(np.concatenate([peaks_pos, peaks_neg]))

t_peaks = t30[pidx]
theta_peaks = signal[pidx]            # estes são centrados (offset subtraído)
amps = np.abs(theta_peaks)

# filtra amplitudes pequenas que atrapalham o log
valid = amps > 1e-6
t_peaks = t_peaks[valid]
amps = amps[valid]
theta_peaks = theta_peaks[valid]

if len(amps) < 3:
    print("AVISO: menos que 3 picos válidos detectados — resultados podem não ser confiáveis.")
    
# ---------------------------------------------------------------------------
# linearização e ajuste linear
# ln(amp) = ln(A) - beta * t  => slope = -beta
# ---------------------------------------------------------------------------
y_log = np.log(amps)
slope, intercept, r_value, p_value, std_err = linregress(t_peaks, y_log)
beta_env = -slope

# cálculo de incertezas (para slope/intercept via resíduos)
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
print("\nRESULTADOS (primeiros 30 s):")
print(f"n_picos usados = {n}")
print(f"beta_env = {beta_env:.5f} ± {beta_err:.5f} 1/s")
print(f"A_env    = {A_env:.3f} ± {A_env_err:.3f} graus")
print(f"R (pearson) = {r_value:.4f}")

# ---------------------------------------------------------------------------
# plota: ln(|pico|) vs t com a reta ajustada
# ---------------------------------------------------------------------------
plt.figure(figsize=(9,6))
plt.scatter(t_peaks, y_log, color='red', label='ln(|θ_pico|) (dados)')
t_line = np.linspace(np.min(t_peaks), np.max(t_peaks), 200)
plt.plot(t_line, slope * t_line + intercept, color='blue',
         label=f'Ajuste linear: slope={slope:.5f} -> β={-slope:.5f} 1/s\nR={r_value:.4f}')
plt.xlabel("Tempo (s)")
plt.ylabel("ln(|θ_pico|) (log do ângulo em graus)")
plt.title("Envelope linearizado (0–30 s)")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------------------------------
# opcional: plota sinal original (0-30s) com o envelope reconstruído
# ---------------------------------------------------------------------------
t_model = np.linspace(0, 30, 1000)
env = A_env * np.exp(-beta_env * t_model)
upper_env = env + offset0
lower_env = -env + offset0

plt.figure(figsize=(10,6))
plt.plot(t30, ang30, '.', ms=4, label='sinal (0-30 s)')
plt.plot(t_model, upper_env, '-', lw=2, label='envelope (+) reconstruído')
plt.plot(t_model, lower_env, '-', lw=2, label='envelope (-) reconstruído')
# marca os picos usados (convertendo de volta ao nível original adicionando offset)
plt.scatter(t_peaks, theta_peaks + offset0, s=40, facecolors='none', edgecolors='k', label='picos usados')
plt.xlabel("Tempo (s)")
plt.ylabel("Ângulo (graus)")
plt.title("Sinal e envelope reconstruído (0–30 s)")
plt.legend()
plt.grid(True)
plt.show()
