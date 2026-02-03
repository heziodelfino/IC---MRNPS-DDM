import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from collections import OrderedDict
from pathlib import Path

# =========================
# CARREGAR DATASET (.npz)
# =========================
DATA_FILE = Path("pendulo_subcritico_doisgamma.npz") 

data = np.load(DATA_FILE)

t = data["t"]
x_noisy = data["x_noisy"]

x_clean = data["x_clean"]

A = float(data["A"])
omega0 = float(data["omega0"])
phi0 = float(data["phi0"])

gamma1 = float(data["gamma1"])
gamma2 = float(data["gamma2"])
tcut = float(data["t_switch"])  # corte (ex: 6.60)

# =========================
# MODELO
# =========================
def modelo_sub(t, A, gamma, omega_d, phi):
    return A * np.exp(-gamma*t) * np.cos(omega_d*t + phi)

# =========================
# CONFIG.
# =========================
y_zoom = (-0.5, 0.5)  # se quiser automático, coloca None

WINDOWS = [
    # Zoom 2 (início)
    dict(
        key="z2",
        tmin=float(t.min()),
        tmax=tcut - 0.01,
        gamma=gamma1,
        inset=dict(loc="lower left", mark=(1, 3)),
    ),
    # Zoom 1 (depois do corte)
    dict(
        key="z1",
        tmin=tcut,
        tmax=float(t.max()),
        gamma=gamma2,
        inset=dict(loc="upper right", mark=(2, 4)),
    ),
]

# =========================
# ESTILO
# =========================
plt.rcParams.update({
    'font.family':'Courier New',
    'font.size':20,
    'axes.labelsize':20,
    'xtick.labelsize':20,
    'ytick.labelsize':20,
    'legend.fontsize':16
})

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlabel("t(s)")
ax.set_ylabel(r"$\theta(t)$ (rad)")
ax.set_aspect('equal', adjustable='datalim')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.75)
ax.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.25)
ax.set_axisbelow(True)

# =========================
# DADOS 
# =========================
X = t
Y = x_noisy

ax.scatter(X, Y,
           label="Subamortecido (dados)",
           s=90, color="#9671bd",
           edgecolors="#6a408d",
           linewidths=1.5, zorder=1)

# =========================
# FUNÇÕES UTIL
# =========================
def auto_ylim(y, pad_frac=0.15):
    y_min = np.min(y)
    y_max = np.max(y)
    span = (y_max - y_min) if y_max > y_min else 1.0
    pad = pad_frac * span
    return (y_min - pad, y_max + pad)

def fit_window(Xw, Yw, gamma, omega0):
    # Fit com gamma FIXO, ajustando A, omega_d, phi
    def model_fixed_gamma(tt, A_, omega_d_, phi_):
        return modelo_sub(tt, A_, gamma, omega_d_, phi_)

    p0 = [A, np.sqrt(max(omega0**2 - gamma**2, 1e-12)), 0.0]
    bounds = ([0.0, 0.0, -2*np.pi], [np.inf, 10.0, 2*np.pi])

    popt, _ = curve_fit(model_fixed_gamma, Xw, Yw,
                        p0=p0, bounds=bounds, maxfev=20000)
    return model_fixed_gamma, popt

def fmt_interval(tmin, tmax):
    return f"{tmin:.2f}–{tmax:.2f}s"

# =========================
# FIT + PLOT (principal e insets)
# =========================
fits = {}

for w in WINDOWS:
    tmin, tmax, gamma = w["tmin"], w["tmax"], w["gamma"]
    mask = (X >= tmin) & (X <= tmax)

    model, popt = fit_window(X[mask], Y[mask], gamma, omega0)

    xp = np.linspace(X[mask].min(), X[mask].max(), 1200)
    yp = model(xp, *popt)

    label_fit = f"FRS ({fmt_interval(tmin, tmax)})"
    ax.plot(xp, yp, color="#422758", ls='-.', linewidth=2.5, zorder=2, label=label_fit)

    fits[w["key"]] = dict(mask=mask, xp=xp, yp=yp, gamma=gamma, tmin=tmin, tmax=tmax, inset=w["inset"])

# =========================
# LEGENDA (AUTO)
# =========================
handles, labels = ax.get_legend_handles_labels()
uniq = OrderedDict()
for h, l in zip(handles, labels):
    uniq[l] = h

n = len(uniq)
ncol = min(3, n)
rows = int(np.ceil(n / ncol))
y_anchor = 1.06 + 0.04*(rows-1)

ax.legend(list(uniq.values()), list(uniq.keys()),
          loc='upper center',
          bbox_to_anchor=(0.5, y_anchor),
          ncol=ncol,
          frameon=False)

# =========================
# INSETS (AUTO)
# =========================
for key, info in fits.items():
    axins = inset_axes(ax, width="60%", height="35%",
                       loc=info["inset"]["loc"], borderpad=1)

    m = info["mask"]
    axins.scatter(X[m], Y[m], s=30,
                  color="#9671bd", edgecolors="#6a408d", linewidths=0.8)

    axins.plot(info["xp"], info["yp"], color="#422758", linewidth=2.5, ls='-.')

    axins.set_xlim(info["tmin"], info["tmax"])
    if y_zoom is None:
        axins.set_ylim(*auto_ylim(Y[m]))
    else:
        axins.set_ylim(*y_zoom)

    axins.set_xticks([])
    axins.set_yticks([])
    axins.text(0.5, 0.05, rf'$\gamma = {info["gamma"]}$',
               transform=axins.transAxes,
               ha="center", va="bottom", fontsize=12, color='black')

    loc1, loc2 = info["inset"]["mark"]
    mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="0.3")

plt.tight_layout()
plt.show()
