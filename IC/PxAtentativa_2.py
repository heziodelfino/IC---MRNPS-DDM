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
# converte para graus se estiver em rad
# ---------------------------------------------------------------------------
if np.nanmax (ang_raw) > 2 * np.pi:
    ang_deg = ang_raw.copy()
    print("Assumindo dados já em graus (não converti).")
else:
    ang_deg = np.degrees(ang_raw)
    print("Convertemos rad -> graus.")

# ---------------------------------------------------------------------------
# Encontrar picos experimentais
# ---------------------------------------------------------------------------
peaks, _ = find_peaks(ang_deg)
t_peaks = t[peaks]
amps = ang_deg[peaks]

# ---------------------------------------------------------------------------
# Calcular períodos e amplitudes
# ---------------------------------------------------------------------------
if len(t_peaks) > 2:
    periods = np.diff(t_peaks)
    amps_used = amps[:-1]   # amplitude no início de cada ciclo
    
else:
    raise RuntimeError("Poucos picos encontrados para calcular períodos.")



# ---------------------------------------------------------------------------
# Ajuste quadrático T(A)
# ---------------------------------------------------------------------------
coeffs = np.polyfit(amps_used, periods, 2)
a2, a1, a0 = coeffs

print("\nAjuste quadrático T(A) = a2*A^2 + a1*A + a0")
print(f"a2 = {a2:.6e}, a1 = {a1:.6e}, a0 = {a0:.6f}")

# ---------------------------------------------------------------------------
# Curva ajustada
# ---------------------------------------------------------------------------
A_fit = np.linspace(np.min(amps_used), np.max(amps_used), 400)
T_fit = a2*A_fit**2 + a1*A_fit + a0

# ---------------------------------------------------------------------------
# Plot bonito
# ---------------------------------------------------------------------------
plt.figure(figsize=(9,6))
plt.scatter(amps_used, periods, s=40, color="#1f1f1f", marker='^', label="Dados experimentais")
plt.plot(A_fit, T_fit, color="#ff8800",
         label=fr"Ajuste: $T(\theta_{0}) = ({a2:.2e})\theta_{0}^2 + ({a1:.2e})\theta_{0} + ({a0:.4f})$")
plt.xlabel(r"Amplitude $\theta_0$ (°)", fontsize=16)
plt.ylabel(r"Período $T$ (s)", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.show()

# from scipy.datasets import electrocardiogram

# x = electrocardiogram()[2000:4000]
# peaks, _ = find_peaks(x, height = 0)
# plt.plot(x)
# plt.plot(peaks, x[peaks], 'x', color = 'red')
# plt.plot(np.zeros_like(x), '--', color = 'grey')
# plt.show()

# print(x[peaks])
# print(_)
