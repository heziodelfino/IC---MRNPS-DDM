import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Caminho do arquivo
# ---------------------------------------------------------------------------
caminho = r"C:\Users\Hézio\Downloads\pendulo pasco-longo(in).csv"

# ---------------------------------------------------------------------------
# Leitura dos dados
# ---------------------------------------------------------------------------
try:
    df = pd.read_csv(caminho, sep=";", decimal=",", encoding="utf-8", on_bad_lines="skip")
except UnicodeDecodeError:
    df = pd.read_csv(caminho, sep=";", decimal=",", encoding="latin1", on_bad_lines="skip")
except Exception as e:
    raise RuntimeError(f"Erro ao ler o CSV: {e}")

# ---------------------------------------------------------------------------
# Verifica as primeiras linhas para entender o formato
# ---------------------------------------------------------------------------
print("Pré-visualização do arquivo:")
print(df.head())
print(f"\nColunas detectadas: {list(df.columns)}")

# ---------------------------------------------------------------------------
# Seleciona automaticamente as duas primeiras colunas numéricas
# ---------------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()

if len(numeric_cols) < 2:
    df = df.apply(pd.to_numeric, errors="coerce")
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()

if len(numeric_cols) < 2:
    raise RuntimeError("Não foi possível identificar duas colunas numéricas no CSV.")

t = df[numeric_cols[0]].to_numpy(dtype=float)
ang_raw = df[numeric_cols[1]].to_numpy(dtype=float)

# ---------------------------------------------------------------------------
# Conversão automática rad → graus, se necessário
# ---------------------------------------------------------------------------
if np.nanmax(np.abs(ang_raw)) > 2 * np.pi:
    ang_deg = ang_raw.copy()
    print("Assumindo dados já em graus (não converti).")
else:
    ang_deg = np.degrees(ang_raw)
    print("Convertemos rad -> graus.")

# ---------------------------------------------------------------------------
# Máscara para recorte temporal
# ---------------------------------------------------------------------------
mask = (t >= 150) & (t <= 1500)
t_fit = t[mask]
ang_fit = ang_deg[mask]

if len(t_fit) < 10:
    raise RuntimeError("Poucos pontos no intervalo 50–1500 s — verifique o arquivo ou o corte de tempo.")

print(f"\nArquivo lido com sucesso: {len(t_fit)} pontos válidos.")

# ----------------------------------------------------------------------------------------------------
# Vars conhecidas (T0 e Beta)
# ----------------------------------------------------------------------------------------------------
T0 = 1.9
beta_known = 0.00125

# ----------------------------------------------------------------------------------------------------
# Modelo teórico com T expandido
# ----------------------------------------------------------------------------------------------------
def damped_model_Tk_expanded(t, A, delta, offset, beta=beta_known, T0=T0):
    """
    Modelo de pêndulo amortecido com período dependente da amplitude,
    usando a expansão até 2ª ordem da integral elíptica completa K(k).

    θ(t) = A e^{-βt} cos( (2πt) / T(t) + δ ) + offset
    """
    # amplitude instantânea
    theta_t = A * np.exp(-beta * t)

    # k(\theta)
    k = np.sin(np.deg2rad(theta_t) / 2)
    k = np.clip(k, -0.9999, 0.9999)

    # expansão até 2ª ordem
    correction = (1
                  + (1/4) * k**2
                  + (9/64) * k**4)

    # período dependente da amplitude
    T_t = T0 * correction
    omega_t = 2 * np.pi / T_t

    return A * np.exp(-beta * t) * np.cos(omega_t * t + delta) + offset

# ----------------------------------------------------------------------------------------------------
# Ajuste
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Preparar chute inicial e bounds adaptativos (robustos)
# ----------------------------------------------------------------------------------------------------
# garantir sem NaNs
ang_clean = ang_fit.copy()
t_clean = t_fit.copy()
mask_ok = np.isfinite(ang_clean) & np.isfinite(t_clean)
ang_clean = ang_clean[mask_ok]
t_clean = t_clean[mask_ok]

if len(ang_clean) < 5:
    raise RuntimeError("Poucos pontos válidos para ajuste após limpeza.")

A_guess = np.nanmax(np.abs(ang_clean))
offset_guess = np.nanmean(ang_clean)
delta_guess = 0.0

A_min = 0.0
A_max = max(1.0, 1.5 * A_guess)        
delta_min, delta_max = -2*np.pi, 2*np.pi
margin = 0.1 * max(1.0, np.ptp(ang_clean))
offset_min = np.nanmin(ang_clean) - margin
offset_max = np.nanmax(ang_clean) + margin

p0 = [A_guess, delta_guess, offset_guess]
lower = [A_min, delta_min, offset_min]
upper = [A_max, delta_max, offset_max]
bounds = (lower, upper)

print("Chute inicial p0:", p0)
print("Bounds lower:", lower)
print("Bounds upper:", upper)

# ---------------------------------------------------------------------------------------
# Tenta o ajuste, com fallback que relaxa bounds se necessário
# ---------------------------------------------------------------------------------------
from scipy.optimize import OptimizeWarning
import warnings

try:
    params, cov = curve_fit(
        damped_model_Tk_expanded,
        t_clean, ang_clean,
        p0=p0, bounds=bounds, maxfev=40000
    )
except ValueError as e:
    print("Aviso: ajuste falhou com bounds iniciais:", e)
    print("Relaxando bounds e tentando novamente...")
    lower_relaxed = [0.0, -10*np.pi, np.nanmin(ang_clean) - 10*np.ptp(ang_clean) - 1.0]
    upper_relaxed = [10*np.max(np.abs(ang_clean)) + 1.0, 10*np.pi, np.nanmax(ang_clean) + 10*np.ptp(ang_clean) + 1.0]
    try:
        params, cov = curve_fit(
            damped_model_Tk_expanded,
            t_clean, ang_clean,
            p0=p0, bounds=(lower_relaxed, upper_relaxed), maxfev=80000
        )
    except Exception as e2:
        # último recurso: sem bounds (pode divergir se modelo ruim)
        print("Relaxamento também falhou:", e2)
        print("Tentando ajuste sem bounds (último recurso)...")
        params, cov = curve_fit(
            damped_model_Tk_expanded,
            t_clean, ang_clean,
            p0=p0, maxfev=100000
        )

# ----------------------------------------------------------------------------------------------------
# Estatísticas do ajuste
# ----------------------------------------------------------------------------------------------------
from scipy.stats import pearsonr

# Predição do modelo com os parâmetros ajustados
ang_pred = damped_model_Tk_expanded(t_clean, *params)

# Resíduos
residuos = ang_clean - ang_pred

R, _ = pearsonr(ang_clean, ang_pred)

# Qui-quadrado e Qui-quadrado reduzido
chi2 = np.sum((residuos) ** 2)
graus_lib = len(ang_clean) - len(params)
chi2_red = chi2 / graus_lib if graus_lib > 0 else np.nan

# Impressão dos resultados no terminal
print("\n=== Estatísticas do Ajuste ===")
print(f"Coeficiente de Pearson (R) = {R:.4f}")
print(f"Qui-quadrado (χ²) = {chi2:.4f}")
print(f"Qui-quadrado reduzido (χ²_red) = {chi2_red:.4f}")


A_fit, delta_fit, offset_fit = params
print(f"Ajuste: A={A_fit:.4f}, δ={delta_fit:.4f}, offset={offset_fit:.4f}")
# ----------------------------------------------------------------------------------------------------
# Gráfico
# ----------------------------------------------------------------------------------------------------
t_model = np.linspace(t_fit.min(), t_fit.max(), 10000)
ang_model = damped_model_Tk_expanded(t_model, *params)

plt.figure(figsize=(10, 5))
plt.plot(t_fit, ang_fit, 'o', ms=4, label='Dados experimentais', alpha=0.7)
plt.plot(t_model, ang_model, '-', lw=2.5, label='Ajuste: modelo expandido', alpha = 0.5)
plt.xlabel("Tempo (s)")
plt.ylabel("Ângulo (°)")
plt.legend()
plt.grid(True, alpha=0.7)
plt.tight_layout()
plt.show()

#---------------------------------------------------------------------------------------
# FIM
#---------------------------------------------------------------------------------------