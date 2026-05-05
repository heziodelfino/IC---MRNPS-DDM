import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Configuração do arquivo e das colunas
# ---------------------------------------------------------------------------
# Deixe serie_3.csv na mesma pasta deste script.
# Se quiser usar outro caminho, altere a linha abaixo.
CAMINHO = Path(__file__).with_name("serie_3.csv")

COLUNA_TEMPO = "tempo_s"
COLUNA_VELOCIDADE_ANGULAR = "velocidade_angular_rad_s"
COLUNA_ANGULO = "angulo_rad"

# A Série 3 extraída vai aproximadamente de 0.05 s até 244.80 s.
# Use None para pegar automaticamente o início/fim disponível.
TEMPO_INICIO = None
TEMPO_FIM = None
PONTOS_MINIMOS = 10
MOSTRAR_GRAFICOS = False  # mude para True se quiser abrir as janelas dos gráficos
PASTA_SAIDA = Path(__file__).parent / "figuras_serie_3"


def carregar_dados(caminho=CAMINHO):
    """Carrega a Série 3 exportada do Capstone e padroniza colunas numéricas."""
    df = pd.read_csv(caminho)

    colunas_necessarias = [COLUNA_TEMPO, COLUNA_VELOCIDADE_ANGULAR, COLUNA_ANGULO]
    faltando = [col for col in colunas_necessarias if col not in df.columns]
    if faltando:
        raise ValueError(
            f"Colunas não encontradas no CSV: {faltando}\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )

    for col in colunas_necessarias:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[COLUNA_TEMPO, COLUNA_VELOCIDADE_ANGULAR, COLUNA_ANGULO])
    df = df.sort_values(COLUNA_TEMPO).reset_index(drop=True)
    return df


def recortar_intervalo(df):
    """Aplica a máscara temporal, usando os limites reais quando None."""
    tempo_min = df[COLUNA_TEMPO].min()
    tempo_max = df[COLUNA_TEMPO].max()

    inicio = tempo_min if TEMPO_INICIO is None else TEMPO_INICIO
    fim = tempo_max if TEMPO_FIM is None else TEMPO_FIM

    mask = (df[COLUNA_TEMPO] >= inicio) & (df[COLUNA_TEMPO] <= fim)
    df_plot = df.loc[mask].copy()

    if len(df_plot) < PONTOS_MINIMOS:
        raise RuntimeError(f"Poucos pontos no intervalo {inicio}s–{fim}s.")

    return df_plot, inicio, fim


def estimar_distancia_picos(t_win, signal):
    """Estima o período dominante via FFT para definir distance no find_peaks."""
    dt = np.median(np.diff(t_win))
    if dt <= 0 or np.isnan(dt):
        dt = 1.0

    n = len(t_win)
    yf = np.fft.rfft(signal - np.mean(signal))
    xf = np.fft.rfftfreq(n, dt)

    if len(xf) > 1:
        idx = np.argmax(np.abs(yf)[1:]) + 1
        f0 = xf[idx]
        period_est = 1.0 / f0 if f0 > 0 else None
    else:
        period_est = None

    if period_est is not None:
        distance_samples = max(3, int(0.6 * period_est / dt))
        print(f"Período estimado via FFT: {period_est:.3f} s.")
        print(f"Distância mínima entre picos: {distance_samples} amostras.")
    else:
        distance_samples = 10
        print(f"Não foi possível estimar o período. Usando {distance_samples} amostras.")

    return distance_samples


def detectar_picos_velocidade(t_win, signal, distance_samples):
    """Detecta picos positivos da velocidade angular."""
    amplitude_range = np.max(signal) - np.min(signal)
    prominence = max(amplitude_range * 0.05, 1e-3)

    peaks_pos, _ = find_peaks(signal, distance=distance_samples, prominence=prominence)

    t_peaks = t_win[peaks_pos]
    vel_peaks = signal[peaks_pos]

    print(f"✅ {len(t_peaks)} picos positivos de velocidade detectados.")
    return peaks_pos, t_peaks, vel_peaks


def detectar_amplitudes_por_ciclo(t_win, angle, t_peaks, distance_samples):
    """
    Detecta picos positivos do ângulo e associa cada pico de velocidade
    ao pico de amplitude mais próximo no tempo.

    Isso é mais seguro do que simplesmente cortar os arrays pelo menor tamanho,
    porque os picos de ângulo e velocidade não acontecem exatamente no mesmo instante.
    """
    peaks_ang, _ = find_peaks(angle, distance=distance_samples)

    if len(peaks_ang) == 0 or len(t_peaks) == 0:
        return np.array([]), np.array([])

    t_ang = t_win[peaks_ang]
    amp_deg = np.degrees(angle[peaks_ang])

    amp_associada = []
    indices_validos = []

    for i, tp in enumerate(t_peaks):
        j = np.argmin(np.abs(t_ang - tp))
        amp_associada.append(amp_deg[j])
        indices_validos.append(i)

    return np.array(amp_associada), np.array(indices_validos, dtype=int)


def finalizar_figura(nome_arquivo):
    """Salva a figura e, opcionalmente, mostra na tela."""
    PASTA_SAIDA.mkdir(exist_ok=True)
    caminho_figura = PASTA_SAIDA / nome_arquivo
    plt.tight_layout()
    plt.savefig(caminho_figura, dpi=300)
    print(f"Figura salva em: {caminho_figura}")

    if MOSTRAR_GRAFICOS:
        finalizar_figura("velocidade_maxima_vs_tempo_serie3.png")
    else:
        plt.close()


def main():
    df = carregar_dados()
    df_plot, inicio, fim = recortar_intervalo(df)

    t_win = df_plot[COLUNA_TEMPO].to_numpy()
    signal = df_plot[COLUNA_VELOCIDADE_ANGULAR].to_numpy()

    distance_samples = estimar_distancia_picos(t_win, signal)
    _, t_peaks, vel_peaks = detectar_picos_velocidade(t_win, signal, distance_samples)

    plt.figure(figsize=(10, 6))

    plt.plot(t_peaks, vel_peaks, "o-", markersize=6)

    plt.title(f"Velocidade máxima vs tempo — Série 3 ({inicio:.2f}s a {fim:.2f}s)", fontsize=16)
    plt.xlabel("Tempo (s)", fontsize=14)
    plt.ylabel("Velocidade máxima (rad/s)", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.7)
    finalizar_figura("Vmaxtempo.png")


if __name__ == "__main__":
    main()
