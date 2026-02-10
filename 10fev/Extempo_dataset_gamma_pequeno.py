# gráfico de energia x tempo (1 regime de amortecimento)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress

# ============================================
# CONFIG
# ============================================
FILE = "pendulo_subcritico_gamma_001.npz"

m = 1.0

# janela única (usa tudo)
T_MIN, T_MAX = 0.0, None

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
    "legend.fontsize": 11
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
    n = len(y)
    w = int(max(3, min(w, n if n % 2 == 1 else n-1)))
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")

def fit_gamma_piece(t, E, tmin, tmax, *, prom_frac, dist_s, smooth_win, stride, cut_start):
    """
    Ajusta ln(E_peak)=intercept+slope*t na janela [tmin,tmax].
    Retorna: gamma, slope, intercept, r2, method, t_peaks, E_peaks
    """
    if tmax is None:
        mask = (t >= tmin)
    else:
        mask = (t >= tmin) & (t <= tmax)

    ts = t[mask]
    Es = E[mask]

    good0 = np.isfinite(ts) & np.isfinite(Es)
    ts = ts[good0]
    Es = Es[good0]
    if len(ts) < 5:
        raise RuntimeError("Janela muito curta / poucos pontos.")

    dt = np.median(np.diff(ts))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt inválido dentro da janela (t está estranho).")

    # ---- FIT 1: peaks
    E_range = np.nanmax(Es) - np.nanmin(Es)
    prom = prom_frac * E_range if np.isfinite(E_range) and E_range > 0 else None

    if dist_s is None:
        min_dist = max(3, int(0.2 / dt))
    else:
        min_dist = max(3, int(dist_s / dt))

    peaks, _ = find_peaks(Es, distance=min_dist, prominence=prom)
    t_peaks = ts[peaks]
    E_peaks = Es[peaks]

    good = np.isfinite(t_peaks) & np.isfinite(E_peaks) & (E_peaks > 0)
    t_peaks = t_peaks[good]
    E_peaks = E_peaks[good]

    method = "peaks"

    # ---- FIT 2: fallback
    if len(t_peaks) < 2:
        method = "smooth+stride"

        cut = int(cut_start * len(ts))
        ts2 = ts[cut:]
        Es2 = Es[cut:]

        if len(ts2) < 5:
            raise RuntimeError("Janela pequena demais após CUT_START.")

        Es_sm = moving_average(Es2, smooth_win)
        Es_sm = np.where(Es_sm > 0, Es_sm, np.nan)

        t_peaks = ts2[::stride]
        E_peaks = Es_sm[::stride]

        ok = np.isfinite(t_peaks) & np.isfinite(E_peaks) & (E_peaks > 0)
        t_peaks = t_peaks[ok]
        E_peaks = E_peaks[ok]

    if len(t_peaks) < 2:
        raise RuntimeError("Sem pontos suficientes pra ajustar ln(E) nessa janela.")

    lnE = np.log(E_peaks)
    slope, intercept, r_value, _, _ = linregress(t_peaks, lnE)
    gamma = -slope / 2
    r2 = r_value**2

    return gamma, slope, intercept, r2, method, t_peaks, E_peaks


# ============================================
# CARREGAR DADOS
# ============================================
data = np.load(FILE)

t = data["t"].astype(float)
x_clean = data["x_clean"].astype(float)

# usa velocidade já salva (melhor que gradient)
if "omega_clean" in data.files:
    v = data["omega_clean"].astype(float)
else:
    v = np.gradient(x_clean, t)

# pega omega0 do arquivo e define k = m*omega0^2 (coerente)
omega0 = float(data["omega0"])
k = m * omega0**2

# sanity + ordenação
mask = np.isfinite(t) & np.isfinite(x_clean) & np.isfinite(v)
t = t[mask]
x_clean = x_clean[mask]
v = v[mask]

if np.any(np.diff(t) <= 0):
    idx = np.argsort(t)
    t = t[idx]
    x_clean = x_clean[idx]
    v = v[idx]

# ============================================
# ENERGIA
# ============================================
T = 0.5 * m * v**2
U = 0.5 * k * x_clean**2
E = T + U

# ============================================
# AJUSTE (1 JANELA)
# ============================================
gamma_fit, slope, intercept, r2, method, tpk, Epk = fit_gamma_piece(
    t, E, T_MIN, T_MAX,
    prom_frac=PROM_FRAC, dist_s=DIST_S,
    smooth_win=SMOOTH_WIN, stride=STRIDE, cut_start=CUT_START
)

gamma_true = float(data["gamma"]) if "gamma" in data.files else np.nan

print("===== SUBCRÍTICO (1 REGIME) =====")
print(f"Arquivo         = {FILE}")
print(f"Janela          = [{T_MIN:.2f}, {t.max():.2f}] s")
print(f"Metodo usado    = {method}")
print(f"N pontos fit    = {len(tpk)}")
print(f"Gamma extraído  = {gamma_fit:.6f}")
if np.isfinite(gamma_true):
    print(f"Gamma do arquivo= {gamma_true:.6f}")
print(f"R² do ajuste    = {r2:.6f}")

# envelope
E_env = np.full_like(t, np.nan, dtype=float)
mask_env = (t >= T_MIN) if (T_MAX is None) else ((t >= T_MIN) & (t <= T_MAX))
E_env[mask_env] = np.exp(intercept) * np.exp(-2 * gamma_fit * t[mask_env])

# ============================================
# LABELS
# ============================================
label_Einst = r"Energia instantânea: $E(t)=\frac{1}{2}mv^2+\frac{1}{2}kx^2$"
label_U = r"Energia Potencial: $U(t) = \frac{1}{2}kx^2$"
label_T = r"Energia Cinética: $T(t) = \frac{1}{2}m\dot{x}^2$"

label_env = rf"Envelope: $E_0 e^{{-2\gamma t}}$  ($\gamma={gamma_fit:.4g}$)"
label_pts = r"Amostras: $E_{\mathrm{peak}}$"
label_fit = rf"Ajuste: $\ln(E)=\ln(E_0)-2\gamma t$  ($R^2={r2:.4g}$)"

# ============================================
# PLOTS (lado a lado)
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: E(t) ---
ax1.set_xlabel("t(s)")
ax1.set_ylabel("E(t)")
style_axes(ax1)

ax1.plot(t, U, label=label_U, linewidth=2.2, color=COL_VERDE, zorder=2)
ax1.plot(t, E, label=label_Einst, linewidth=2.2, color=COL_AZUL, zorder=2, ls='-.')
ax1.plot(t, T, label=label_T, linewidth=2.2, color=COL_MAGENTA, zorder=2)

ax1.plot(t, E_env, label=label_env, color=COL_GREY, linewidth=2.2, ls="--", zorder=1)

ax1.scatter(tpk, Epk, label=label_pts,
            s=55, color="#E9D8B4", edgecolors="#B98B4E",
            linewidths=1.2, zorder=3)

h1, l1 = ax1.get_legend_handles_labels()
ax1.legend(h1, l1, loc="upper center", bbox_to_anchor=(0.5, 1.22),
           ncol=1, frameon=True)

# --- Plot 2: ln(E) ---
ax2.set_xlabel("t(s)")
ax2.set_ylabel(r"$\ln(E)$")
style_axes(ax2)

lnE = np.log(Epk)
ax2.scatter(tpk, lnE, label=label_fit,
            s=55, color="#E9D8B4", edgecolors="#B98B4E",
            linewidths=1.2, zorder=3)

xp = np.linspace(tpk.min(), tpk.max(), 300)
ax2.plot(xp, intercept + slope * xp, color=COL_GREY, linewidth=2.2, ls="--", zorder=2)

h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h2, l2, loc="upper center", bbox_to_anchor=(0.5, 1.22),
           ncol=1, frameon=True)

plt.tight_layout()
plt.show()
