#usando o dataset do pendulo_subcritico.npz foi gerado um 
#gráfico de energiaxtempo, cumprindo a tarefa dada no dia 16 de jan

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress

# ============================================
# CONFIG
# ============================================
FILE = "pendulo_subcritico.npz"

m = 1.0
k = 3.61 # (k = m*omega0**2 se tiver omega0)

# --- ajustes do fit ---
PROM_FRAC   = 0.02
DIST_S      = None

SMOOTH_WIN  = 301
STRIDE      = 30
CUT_START   = 0.05

# ============================================
# ESTILO 
# ============================================
plt.rcParams.update({
    "font.family": "Courier New",
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13
})

COL_MAGENTA = "#9671bd"
EDGE_MAGENTA = "#6a408d"
COL_AZUL = "#77b5b6"
EDGE_AZUL = "#378d94"
COL_GREY = "#8a8a8a"
COL_VERDE = "#beee62"
EDGE_VERDE = "#70ae6e"

def style_axes(ax):
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.75)
    ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.25)
    ax.set_axisbelow(True)

def moving_average(y, w):
    w = int(max(3, w))
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")

# ============================================
# CARREGAR DADOS
# ============================================
data = np.load(FILE)

t = data["t"]              # tempo
x_clean = data["x_clean"]  # posição

mask = np.isfinite(t) & np.isfinite(x_clean)
t = t[mask]
x_clean = x_clean[mask]

if np.any(np.diff(t) <= 0):
    idx = np.argsort(t)
    t = t[idx]
    x_clean = x_clean[idx]

v = np.gradient(x_clean, t)

# ============================================
# ENERGIA
# ============================================
T = 0.5 * m * v**2
U = 0.5 * k * x_clean**2
E = T + U

# ============================================
# FIT 1: picos
# ln(E_peak) = intercept + slope*t, slope = -2*gamma
# ============================================
E_range = np.nanmax(E) - np.nanmin(E)
prom = PROM_FRAC * E_range if np.isfinite(E_range) and E_range > 0 else None

dt = np.median(np.diff(t))
if not np.isfinite(dt) or dt <= 0:
    raise ValueError("dt inválido: seu vetor de tempo t está estranho.")

if DIST_S is None:
    min_dist = max(3, int(0.2 / dt))
else:
    min_dist = max(3, int(DIST_S / dt))

peaks, _ = find_peaks(E, distance=min_dist, prominence=prom)

t_peaks = t[peaks]
E_peaks = E[peaks]

good = np.isfinite(t_peaks) & np.isfinite(E_peaks) & (E_peaks > 0)
t_peaks = t_peaks[good]
E_peaks = E_peaks[good]

method = "peaks"

# ============================================
# FIT 2: fallback
# ============================================
if len(t_peaks) < 2:
    method = "smooth+stride"

    cut = int(CUT_START * len(t))
    t2 = t[cut:]
    E2 = E[cut:]

    Es = moving_average(E2, SMOOTH_WIN)
    Es = np.where(Es > 0, Es, np.nan)

    t_peaks = t2[::STRIDE]
    E_peaks = Es[::STRIDE]

    ok = np.isfinite(t_peaks) & np.isfinite(E_peaks) & (E_peaks > 0)
    t_peaks = t_peaks[ok]
    E_peaks = E_peaks[ok]

if len(t_peaks) < 2:
    raise RuntimeError("Sem pontos suficientes pra ajustar ln(E).")

lnE = np.log(E_peaks)
slope, intercept, r_value, _, _ = linregress(t_peaks, lnE)
gamma = -slope / 2

print(f"Metodo usado   = {method}")
print(f"N pontos fit   = {len(t_peaks)}")
print(f"Gamma extraído = {gamma:.6f}")
print(f"R² do ajuste   = {r_value**2:.6f}")

E_env_fit = np.exp(intercept) * np.exp(-2* gamma * t) 

# labels com expressão (claros)
label_Einst = r"Energia instantânea: $E(t)=\frac{1}{2}mv^2+\frac{1}{2}kx^2$"
label_U = r"Energia Potencial: $U(t) = \frac{1}{2}kx^2$"
label_T = r"Energia Cinética: $T(t) = \frac{1}{2}m\dot{x}^2$"
label_env   = rf"Envelope exp.: $E_0 e^{{-2\gamma t}}$  ($\gamma={gamma:.4g}$)"
label_pts   = r"Amostras p/ fit: $E_{\mathrm{peak}}(t)$"
label_lnpts = r"Pontos: $\ln(E_{\mathrm{peak}})$"
label_fit   = rf"Ajuste linear: $\ln(E)=\ln(E_0)-2\gamma t$  ($R^2={r_value**2:.4g}$)"

# ============================================
# PLOTS (lado a lado)
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: E(t) ---
ax1.set_xlabel("t(s)")
ax1.set_ylabel("E(t)")
style_axes(ax1)

# ax1.plot(t, T, label=label_Einst, linewidth=2.2, color=COL_AZUL, zorder=2)
ax1.plot(t, U, label=label_U, linewidth=2.2, color=COL_VERDE, zorder=2)
ax1.plot(t, E, label=label_Einst, linewidth=2.2, color=COL_AZUL, zorder=2, ls = '-.')
ax1.plot(t, T, label=label_T, linewidth=2.2, color= COL_MAGENTA, zorder=2)
ax1.plot(t, E_env_fit, label=label_env, color=COL_GREY,
         linewidth=2.2, ls="-.", zorder=1)
ax1.scatter(t_peaks, E_peaks, label=label_pts,
            s=55, color=COL_AZUL, edgecolors=EDGE_AZUL,
            linewidths=1.2, zorder=3)

h1, l1 = ax1.get_legend_handles_labels()
ax1.legend(h1, l1, loc="upper center", bbox_to_anchor=(0.5, 1.18),
           ncol=1, frameon=True)

# --- Plot 2: ln(E) ---
ax2.set_xlabel("t(s)")
ax2.set_ylabel(r"$\ln(E)$")
style_axes(ax2)

ax2.scatter(t_peaks, lnE, label=label_lnpts,
            s=55, color=COL_AZUL, edgecolors=EDGE_AZUL,
            linewidths=1.2, zorder=3)

xp = np.linspace(t_peaks.min(), t_peaks.max(), 400)
ax2.plot(xp, intercept + slope * xp, label=label_fit,
         color=COL_GREY, linewidth=2.2, ls="-.", zorder=2)

h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h2, l2, loc="upper center", bbox_to_anchor=(0.5, 1.18),
           ncol=1, frameon=True)

plt.tight_layout()
plt.show()
