import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ============================================
# CONFIG (EDITE AQUI)
# ============================================
FILE_CSV = r"C:\Users\Hézio\python\graficosyulezin\MCA&IC\24fev\pendulo pasco-longo__in.csv"  # seu CSV experimental
SEP = ";"                                 # PASCO costuma ser ';'
DECIMAL = ","                              # decimal com vírgula

# --- colunas ---
TIME_COL  = None   # ex: "Tempo (s) Série #37"
THETA_COL = None   # ex: "Ângulo (rad) Série #37"
OMEGA_COL = None   # ex: "Velocidade angular (rad/s) Série #37"

# --- parâmetros físicos/escala (para energia tipo oscilador harmônico) ---
m = 1.0
omega0 = 1.9                 # <<< coloque o omega0 que você usa/mediu
k = m * omega0**2            # coerente com theta'' + 2γ theta' + ω0^2 theta = 0

# --- dois regimes (corte) ---
TCUT = 300
T1_MIN, T1_MAX = 0.0, TCUT - 0.01
T2_MIN, T2_MAX = TCUT, None

# --- GAMMAS MANUAIS (VOCÊ EDITA AQUI) ---
GAMMA1_MANUAL = 0.0002890
GAMMA2_MANUAL = 0.0001148

# --- detecção de picos / fallback ---
PROM_FRAC  = 0.02
DIST_S     = None            # None = automático (~0.2s)
SMOOTH_WIN = 301
STRIDE     = 30
CUT_START  = 0.05            # corta 5% do começo da janela no fallback

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
COL_AZUL = "#77b5b6"
COL_GREY = "#8a8a8a"
COL_VERDE = "#beee62"

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

def auto_pick_col(cols, must_contain):
    """Escolhe a primeira coluna que contém TODAS as substrings em must_contain."""
    cols_lower = {c: c.lower() for c in cols}
    want = [s.lower() for s in must_contain]
    for c, cl in cols_lower.items():
        if all(w in cl for w in want):
            return c
    return None

def load_experiment_csv(path):
    df = pd.read_csv(path, sep=SEP, decimal=DECIMAL, encoding="utf-8-sig")

    cols = list(df.columns)

    time_col  = TIME_COL  or auto_pick_col(cols, ["tempo", "(s)"])
    theta_col = THETA_COL or auto_pick_col(cols, ["ângulo", "(rad)"])
    omega_col = OMEGA_COL or auto_pick_col(cols, ["velocidade", "angular", "rad/s"])

    if time_col is None or theta_col is None:
        raise ValueError(
            "Não consegui achar automaticamente as colunas de tempo e ângulo.\n"
            f"Colunas disponíveis: {cols}\n"
            "Defina TIME_COL e THETA_COL manualmente no CONFIG."
        )

    t = df[time_col].to_numpy(dtype=float)
    theta = df[theta_col].to_numpy(dtype=float)

    omega_raw = None
    if omega_col is not None:
        omega_raw = df[omega_col].to_numpy(dtype=float)

    # limpa/sorta
    mask = np.isfinite(t) & np.isfinite(theta)
    t = t[mask]
    theta = theta[mask]
    omega_raw = omega_raw[mask] if omega_raw is not None else None

    if np.any(np.diff(t) <= 0):
        idx = np.argsort(t)
        t = t[idx]
        theta = theta[idx]
        omega_raw = omega_raw[idx] if omega_raw is not None else None

    # omega: usa coluna se for boa; senão gradient
    omega_grad = np.gradient(theta, t)

    if omega_raw is None:
        omega = omega_grad
        omega_source = "gradient(theta)"
    else:
        frac_ok = np.mean(np.isfinite(omega_raw))
        # se tiver muitos NaN, troca tudo por gradient
        if frac_ok < 0.8:
            omega = omega_grad
            omega_source = "gradient(theta) (omega_csv ruim)"
        else:
            omega = omega_raw.copy()
            bad = ~np.isfinite(omega)
            omega[bad] = omega_grad[bad]
            omega_source = "omega_csv (NaNs preenchidos com gradient)"

    return t, theta, omega, omega_source, (time_col, theta_col, omega_col)

def pick_peak_samples(ts, Es, prom_frac, dist_s, smooth_win, stride, cut_start):
    """Retorna (method, t_peaks, E_peaks)."""
    dt = np.median(np.diff(ts))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt inválido na janela.")

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
        raise RuntimeError("Sem pontos suficientes pra envelope nessa janela.")

    return method, t_peaks, E_peaks

def fit_intercept_fixed_gamma(t_peaks, E_peaks, gamma):
    """
    Com slope fixo = -2*gamma:
        ln(E) = intercept + slope*t
    Estima intercept robusto (mediana) e calcula R² do ajuste com slope fixo.
    """
    lnE = np.log(E_peaks)
    slope = -2.0 * gamma

    intercept = np.nanmedian(lnE - slope * t_peaks)  # equivalente a median(lnE + 2γt)

    yhat = intercept + slope * t_peaks
    ss_res = np.nansum((lnE - yhat) ** 2)
    ss_tot = np.nansum((lnE - np.nanmean(lnE)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return slope, intercept, r2

def window_slice(t, y, tmin, tmax):
    if tmax is None:
        mask = (t >= tmin)
    else:
        mask = (t >= tmin) & (t <= tmax)
    return t[mask], y[mask], mask

# ============================================
# MAIN
# ============================================
t, theta, omega, omega_source, (time_col, theta_col, omega_col) = load_experiment_csv(FILE_CSV)

print("===== CSV LIDO =====")
print(f"Arquivo          = {FILE_CSV}")
print(f"Coluna tempo     = {time_col}")
print(f"Coluna ângulo    = {theta_col}")
print(f"Coluna omega     = {omega_col}")
print(f"Omega usado      = {omega_source}")
print(f"n pontos         = {len(t)}")
print()

# energia (modelo harmônico equivalente)
T = 0.5 * m * omega**2
U = 0.5 * k * theta**2
E = T + U

# --- janela 1 ---
t1, E1, mask1 = window_slice(t, E, T1_MIN, T1_MAX)
method1, tpk1, Epk1 = pick_peak_samples(
    t1, E1,
    prom_frac=PROM_FRAC, dist_s=DIST_S,
    smooth_win=SMOOTH_WIN, stride=STRIDE, cut_start=CUT_START
)
slope1, intercept1, r2_1 = fit_intercept_fixed_gamma(tpk1, Epk1, GAMMA1_MANUAL)

# --- janela 2 ---
t2, E2, mask2 = window_slice(t, E, T2_MIN, T2_MAX)
method2, tpk2, Epk2 = pick_peak_samples(
    t2, E2,
    prom_frac=PROM_FRAC, dist_s=DIST_S,
    smooth_win=SMOOTH_WIN, stride=STRIDE, cut_start=CUT_START
)
slope2, intercept2, r2_2 = fit_intercept_fixed_gamma(tpk2, Epk2, GAMMA2_MANUAL)

print("===== REGIME 1 (gamma fixo) =====")
print(f"Janela        = [{T1_MIN:.2f}, {T1_MAX:.2f}] s")
print(f"Metodo        = {method1}")
print(f"N pontos      = {len(tpk1)}")
print(f"gamma_1 manual= {GAMMA1_MANUAL:.6f}")
print(f"R² (slope fixo)= {r2_1:.6f}")
print()

print("===== REGIME 2 (gamma fixo) =====")
print(f"Janela        = [{T2_MIN:.2f}, {t.max():.2f}] s")
print(f"Metodo        = {method2}")
print(f"N pontos      = {len(tpk2)}")
print(f"gamma_2 manual= {GAMMA2_MANUAL:.6f}")
print(f"R² (slope fixo)= {r2_2:.6f}")
print()

# envelopes por partes
E_env_1 = np.full_like(t, np.nan, dtype=float)
E_env_2 = np.full_like(t, np.nan, dtype=float)

E_env_1[mask1] = np.exp(intercept1) * np.exp(-2 * GAMMA1_MANUAL * t[mask1])
E_env_2[mask2] = np.exp(intercept2) * np.exp(-2 * GAMMA2_MANUAL * t[mask2])

# ============================================
# LABELS
# ============================================
label_Einst = r"Energia instantânea: $E(t)=\frac{1}{2}m\dot{\theta}^2+\frac{1}{2}k\theta^2$"
label_U = r"Potencial: $U(t)=\frac{1}{2}k\theta^2$"
label_T = r"Cinética: $T(t)=\frac{1}{2}m\dot{\theta}^2$"

label_env1 = rf"Envelope 1 (fixo): $E_0 e^{{-2\gamma_1 t}}$  ($\gamma_1={GAMMA1_MANUAL:.4g}$)"
label_env2 = rf"Envelope 2 (fixo): $E_0 e^{{-2\gamma_2 t}}$  ($\gamma_2={GAMMA2_MANUAL:.4g}$)"

label_fit1 = rf"Ajuste reg.1: $\ln(E)=\ln(E_0)-2\gamma_1 t$  ($R^2={r2_1:.4g}$)"
label_fit2 = rf"Ajuste reg.2: $\ln(E)=\ln(E_0)-2\gamma_2 t$  ($R^2={r2_2:.4g}$)"

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

ax1.plot(t, E_env_1, label=label_env1, color=COL_GREY, linewidth=2.2, ls="-.", zorder=1)
ax1.plot(t, E_env_2, label=label_env2, color=COL_GREY, linewidth=2.2, ls="--", zorder=1)

ax1.scatter(tpk1, Epk1, label="Picos reg.1",
            s=55, color="#E9D8B4", edgecolors="#B98B4E",
            linewidths=1.2, zorder=3)

ax1.scatter(tpk2, Epk2, label="Picos reg.2",
            s=55, marker="s", color="#E9D8B4", edgecolors="#B98B4E",
            linewidths=1.2, zorder=3)

ax1.axvline(TCUT, color="0.3", lw=1.2, ls=":", zorder=0)

h1, l1 = ax1.get_legend_handles_labels()
ax1.legend(h1, l1, loc="upper center", bbox_to_anchor=(0.5, 1.22),
           ncol=1, frameon=True)

# --- Plot 2: ln(E) ---
ax2.set_xlabel("t(s)")
ax2.set_ylabel(r"$\ln(E)$")
style_axes(ax2)

lnE1 = np.log(Epk1)
lnE2 = np.log(Epk2)

ax2.scatter(tpk1, lnE1, label=label_fit1,
            s=55, color="#E9D8B4", edgecolors="#B98B4E",
            linewidths=1.2, zorder=3)

ax2.scatter(tpk2, lnE2, label=label_fit2,
            s=55, marker="s", color="#E9D8B4", edgecolors="#B98B4E",
            linewidths=1.2, zorder=3)

xp1 = np.linspace(tpk1.min(), tpk1.max(), 300)
xp2 = np.linspace(tpk2.min(), tpk2.max(), 300)

ax2.plot(xp1, intercept1 + slope1 * xp1, color=COL_GREY, linewidth=2.2, ls="-.", zorder=2)
ax2.plot(xp2, intercept2 + slope2 * xp2, color=COL_GREY, linewidth=2.2, ls="--", zorder=2)

ax2.axvline(TCUT, color="0.3", lw=1.2, ls=":", zorder=0)

h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h2, l2, loc="upper center", bbox_to_anchor=(0.5, 1.22),
           ncol=1, frameon=True)

plt.tight_layout()
plt.show()