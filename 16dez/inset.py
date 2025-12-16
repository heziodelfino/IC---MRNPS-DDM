import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""
graph color pallete:
[MAGENTA]
#9671bd
#6a408d

[AZUL]
#378d94
#77b5b6
#205458

[VERDE]
#beee62
#70ae6e
#375736
"""

# ------------------ PEQUENAS AMPLITUDES ------------------

A = 0.2         # pequena amplitude (regime linear)
omega0 = 1.9    # frequência natural (ok)
phi0 = 0.0       # solto do repouso
t = np.linspace(0, 20, 1000)

#-------------------------------------------------------------------------
# ------------------------- [Subcrítico] ------------------------------
#-------------------------------------------------------------------------
gamma_sub = 0.05
omega_d = np.sqrt(omega0**2 - gamma_sub**2)
x_sub = A * np.exp(-gamma_sub*t) * np.cos(omega_d * t)
#-------------------------------------------------------------------------
# ---------------------------- [Crítico] ---------------------------------
#-------------------------------------------------------------------------
gamma_crit = omega0
x_crit = A * np.exp(-gamma_crit*t) * (1 + gamma_crit*t)
#-------------------------------------------------------------------------
# --------------------------- [Supercrítico] --------------------------
#-------------------------------------------------------------------------
theta0 = 0.1

gamma_super = 3.0
alpha = np.sqrt(gamma_super**2 - omega0**2)

C1 = (gamma_super - alpha)/(2*alpha) * theta0
C2 = (gamma_super + alpha)/(2*alpha) * theta0

x_super = (C1 * np.exp(-(gamma_super + alpha)*t) +
           C2 * np.exp(-(gamma_super - alpha)*t))


#-------------------------------------------------------------------------
# ------------------------- [Dataset com ruído] --------------------------
#-------------------------------------------------------------------------
np.random.seed(0)
noise_level = 0.05

x_noise0 = x_sub  + noise_level*np.random.normal(size=len(t))
x_noise1 = x_super + noise_level*np.random.normal(size=len(t))
x_noise2 = x_crit + noise_level*np.random.normal(size=len(t))

#-------------------------------------------------------------------------

from scipy.optimize import curve_fit

# ------------------ MODELOS ------------------

def modelo_sub(t, A, gamma, omega_d, phi):
    return A * np.exp(-gamma*t) * np.cos(omega_d*t + phi)

def modelo_crit(t, A, gamma):
    return A * np.exp(-gamma*t) * (1 + gamma*t)

def modelo_super(t, C1, C2, gamma, alpha):
    alpha = np.sqrt(gamma**2 - omega0**2)
    return (C1 * np.exp(-(gamma + alpha)*t) +
            C2 * np.exp(-(gamma - alpha)*t))


# ------------------ MODELOS ------------------


datasets = [
    {
        "name": "Subamortecido",
        "x": t,
        "y": x_noise0,
        "color": "#9671bd",
        "edge": "#6a408d",
        "color_fit": "#422758",
        "model": "sub"
    },
    {
        "name": "Crítico",
        "x": t,
        "y": x_noise2,
        "color": "#77b5b6",
        "edge": "#378d94",
        "color_fit": "#205458",
        "model": "crit"
    },
    {
        "name": "Super-Crítico",
        "x": t,
        "y": x_noise1,
        "color": "#beee62",
        "edge": "#70ae6e",
        "color_fit": "#375736",
        "model": "super"
    }
]
# ===== estilo =====
plt.rcParams.update({
    'font.family':'Courier New',
    'font.size':20,
    'axes.labelsize':20,
    'xtick.labelsize':20,
    'ytick.labelsize':20,
    'legend.fontsize':16
})

fig,ax = plt.subplots(figsize=(9,9))

#mude os labels em x e em y de acordo como vc desejar
ax.set_xlabel("t(s)")
ax.set_ylabel(r"$\theta(t)$ (rad)")

# (opcional ajusts)
ax.set_aspect('equal', adjustable='datalim')
#ax.set_aspect('auto')
ax.minorticks_on()
# grid, style e etc vão aqui
ax.grid(True,which='major',linestyle='-',linewidth=0.75,alpha=0.75)
ax.grid(True,which='minor',linestyle='-',linewidth=0.25,alpha=0.25)
ax.set_axisbelow(True)
# para pegar range global
all_x = []
all_y = []

regression_flag = 0

for data in datasets:
    X = data["x"]
    Y = data["y"]

    # scatter
    ax.scatter(X, Y, 
               label=data["name"],
               s=90,
               color=data["color"],
               edgecolors=data["edge"],
               linewidths=1.5,
               zorder=1)

    # linear fit
    # coef = np.polyfit(X,Y,1)
    # p    = np.poly1d(coef)
    # xp   = np.linspace(X.min(),X.max(),200)
    # yp   = p(xp)

    # ax.plot(xp, yp,
    #         label="LR" if regression_flag==0 else "_nolegend_",
    #         color="#8a8a8a",
    #         linewidth=2.5,
    #         ls = '-.',
    #         zorder=2)


    # ---------------------------------------------------
    # FIT FÍSICO por regime
    # ---------------------------------------------------
    xp = np.linspace(X.min(), X.max(), 2000)

    if data["model"] == "sub":
        popt, _ = curve_fit(modelo_sub, X, Y,
                            p0=[1.0, 0.5, 1.5, 0.0])
        yp = modelo_sub(xp, *popt)
        label_fit = "FRS"

    elif data["model"] == "crit":
        popt, _ = curve_fit(modelo_crit, X, Y,
                            p0=[1.0, 2.0])
        yp = modelo_crit(xp, *popt)
        label_fit = "FRC"

    elif data["model"] == "super":
        popt, _ = curve_fit(modelo_super, X, Y,
                            p0=[0.5, 0.5, 3.0, 2.0])
        yp = modelo_super(xp, *popt)
        label_fit = "FRSC"

    ax.plot(xp, yp,
            label=label_fit,
            color = data["color_fit"],
            ls = '-.',
            linewidth=2.5,
            zorder=2)


    all_x.extend(X)
    all_y.extend(Y)
    regression_flag += 1

# ------------------------------------------
# legenda em cima do plot
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels,
          loc='upper center',
          bbox_to_anchor=(0.5,1.10),
          ncol=3.0,
          frameon=False)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# região de zoom (IGUAL para todas)
x1, x2 = 0.0, 5.0
y1, y2 = -0.5, 0.5

# ================= ZOOM 1 — canto superior direito =================
axins1 = inset_axes(
    ax, width="35%", height="35%",
    loc="upper right", borderpad=1
)

axins1.scatter(t, x_noise0, s=30,
               color="#9671bd", edgecolors="#6a408d", linewidths=0.8)
axins1.plot(t, x_sub, color="#422758", linewidth=2.5, ls = '-.')

axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)
axins1.set_xticks([])
axins1.set_yticks([])
axins1.text(0.5, 0.05, r'$\beta = 0.05$', transform=axins1.transAxes, 
            ha="center", va="bottom", fontsize=12, color='black')

mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.3")

# ================= ZOOM 2 — canto inferior direito =================
axins2 = inset_axes(
    ax, width="35%", height="35%",
    loc="lower right", borderpad=1
)

axins2.scatter(t, x_noise1, s=30,
               color="#beee62", edgecolors="#70ae6e", linewidths=0.8)
axins2.plot(t, x_super, color="#375736", linewidth=2.5, ls = '-.')

axins2.set_xlim(x1, x2)
axins2.set_ylim(y1, y2)
axins2.set_xticks([])
axins2.set_yticks([])
axins2.text(0.5, 0.05, rf'$\beta = {gamma_super}$', transform=axins2.transAxes, 
            ha="center", va="bottom", fontsize=12, color='black')

mark_inset(ax, axins2, loc1=1, loc2=3, fc="none", ec="0.3")

# ================= ZOOM 3 — canto inferior esquerdo =================
axins3 = inset_axes(
    ax, width="35%", height="35%",
    loc="lower left", borderpad=1
)

axins3.scatter(t, x_noise2, s=30,
               color="#77b5b6", edgecolors="#378d94", linewidths=0.8)
axins3.plot(t, x_crit, color="#205458", linewidth=2.5, ls = '-.')

axins3.set_xlim(x1, x2)
axins3.set_ylim(y1, y2)
axins3.set_xticks([])
axins3.set_yticks([])
axins3.text(0.5, 0.05, rf'$\beta = \omega_{0} ={gamma_crit}$', transform=axins3.transAxes, 
            ha="center", va="bottom", fontsize=12, color='black')

mark_inset(ax, axins3, loc1=1, loc2=3, fc="none", ec="0.3")


plt.tight_layout()
plt.show()

# ===================== SALVANDO DATASETS =====================

# Subcrítico
np.savez(
    "pendulo_subcritico.npz",
    t=t,
    x_clean=x_sub,
    x_noisy=x_noise0,
    A=A,
    omega0=omega0,
    gamma=gamma_sub
)

# Crítico
np.savez(
    "pendulo_critico.npz",
    t=t,
    x_clean=x_crit,
    x_noisy=x_noise2,
    A=A,
    omega0=omega0,
    gamma=gamma_crit
)

# Supercrítico
np.savez(
    "pendulo_supercritico.npz",
    t=t,
    x_clean=x_super,
    x_noisy=x_noise1,
    theta0=theta0,
    omega0=omega0,
    gamma=gamma_super
)
