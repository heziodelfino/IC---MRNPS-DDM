import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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
# Encontrar picos experimentais (máximos locais)
# ---------------------------------------------------------------------------
peaks, _ = find_peaks(ang_deg)
t_peaks = t[peaks]
amps = ang_deg[peaks]

# ---------------------------------------------------------------------------
# Calcular períodos e amplitudes médias
# ---------------------------------------------------------------------------
if len(t_peaks) > 2:
    periods = np.diff(t_peaks)  # diferença entre picos consecutivos
    amps_mid = (amps[:-1] + amps[1:]) / 2.0  # amplitude média
else:
    raise RuntimeError("Poucos picos encontrados para calcular períodos.")

# ---------------------------------------------------------------------------
# Ajuste quadrático T(A)
# ---------------------------------------------------------------------------
coeffs = np.polyfit(amps_mid, periods, 2)  # [a2, a1, a0]
a2, a1, a0 = coeffs

print("\nAjuste quadrático T(A) = a2*A^2 + a1*A + a0")
print(f"a2 = {a2:.6e}, a1 = {a1:.6e}, a0 = {a0:.6f}")

# ---------------------------------------------------------------------------
# Plot 1: Sinal angular bruto com picos
# ---------------------------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(t, ang_deg, label="θ(t) (graus)")
plt.scatter(t_peaks, amps, color='red', s=20, label="Picos detectados")
plt.xlabel("Tempo (s)")
plt.ylabel("Ângulo (graus)")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------------------------------
# Plot 2: Período vs Amplitude
# ---------------------------------------------------------------------------
plt.figure(figsize=(9,6))
plt.scatter(amps_mid, periods, s=30, color='red', label="Dados (T vs A)")
A_fit = np.linspace(0, np.max(amps_mid)*1.1, 200)
T_fit = a2*A_fit**2 + a1*A_fit + a0
plt.plot(A_fit, T_fit, 'b-', lw=2, label="Ajuste quadrático")

plt.xlabel("Amplitude do pico (graus)")
plt.ylabel("Período T (s)")
plt.legend()
plt.grid(True)
plt.show()
